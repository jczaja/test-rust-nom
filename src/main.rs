// PArsing, find entry block "ENTRY" and then parameters scan for "parameter(" so they are valid
// within this block
// At the end I should now parameters per kernel
//
// sharding={replicated??} check if I can gen info on sharded configuration
// type od data and shape

use nom::{
    bytes::complete::is_a, bytes::complete::tag, bytes::complete::take, bytes::complete::take_till,
    bytes::complete::take_until, bytes::complete::take_while, character::complete::digit1,
    character::is_alphabetic, character::is_digit, error::Error, sequence::delimited,
    sequence::tuple, IResult,
};

fn parser(s: &str) -> IResult<&str, &str> {
    let (s, _) = take_until("Hello")(s)?;
    let k = tag("Hello")(s)?;
    Ok(k)
}

#[derive(Debug, PartialEq)]
enum KernelArgument {
    Typef32(u32, u32),
}

pub fn is_not_char_digit(chr: char) -> bool {
    return chr.is_ascii() == false || is_digit(chr as u8) == false;
}

fn is_arg_type(input: &str) -> bool {
    is_a::<&str, &str, Error<&str>>("f32")(input).is_ok()
}

fn get_type_shape_and_id(input: &str) -> Option<(&str, KernelArgument)> {
    let mut s = input;

    // get next candidate str

    loop {
        // TODO: make alternative of patterns
        let scanner = take_until::<&str, &str, Error<&str>>("f32");

        // Tuple of type[shape]{digits} parameter(id)

        let type_parser = tag::<&str, &str, Error<_>>("f32");

        let shape_parser = delimited(
            tag::<&str, &str, Error<&str>>("["),
            take_till(is_not_char_digit),
            tag("]"),
        );

        let braced_parser = delimited(
            tag::<&str, &str, Error<&str>>("{"),
            take_till(is_not_char_digit),
            tag("}"),
        );

        let parameter_parser = delimited(
            tag::<&str, &str, Error<&str>>("parameter("),
            take_till(is_not_char_digit),
            tag(")"),
        );

        let result = scanner(s);
        s = match result {
            Ok((s, _)) => s,
            Err(_) => return None,
        };

        // matching example: "f32[100]{0} parameter(0)"
        let result = tuple((
            &type_parser,
            shape_parser,
            braced_parser,
            take(1usize),
            parameter_parser,
        ))(s);

        match result {
            Ok((s, (arg_type, shape_str, _, _, id_str))) => {
                let shape =
                    str::parse::<u32>(shape_str).expect("Error: converting shape to int failed");

                let arg_id = str::parse::<u32>(id_str).expect("Error: converting to int failed");

                let a = match arg_type {
                    "f32" => Some((s, KernelArgument::Typef32(shape, arg_id))),
                    _ => panic!("Error: Unsupported argument type: {arg_type}"),
                };
                return a;
            }
            Err(_) => {
                let (remainder, _) = type_parser(s).expect("Error: parsing type");
                s = remainder;
            }
        }
    }
}

fn get_entry_block(input: &str) -> &str {
    let (s, _) =
        take_until::<&str, &str, Error<&str>>("ENTRY")(input).expect("error: no ENTRY in input!!");

    let (s, _) =
        take_until::<&str, &str, Error<&str>>("{")(s).expect("Error: No parameter in input!!");
    let nesting_level = std::cell::Cell::new(0);

    // scan characters and either increase nesting or try if is_argtype
    let result: Result<(&str, &str), nom::Err<Error<_>>> = take_while(|c: char| {
        if c == '}' {
            nesting_level.set(nesting_level.get() - 1);
        } else if c == '{' {
            nesting_level.set(nesting_level.get() + 1);
        }
        c != '}' || nesting_level.get() != 0
    })(s);
    let (_, block) = result.expect("Error: ill-defined ENTRY block");
    block
}

fn get_args_desc(input: &str) -> Vec<KernelArgument> {
    let mut arguments: Vec<KernelArgument> = vec![];

    let block = get_entry_block(input);
    let mut s = block;
    loop {
        let maybe_arg = get_type_shape_and_id(s);
        s = match maybe_arg {
            Some((s, arg_desc)) => {
                arguments.push(arg_desc);
                s
            }
            None => break,
        };
    }

    arguments
}

fn main() {
    println!("Hello Rust Parsers world!");

    let input = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/module_0002.jit__logsm_from_logmhalo_jax_kern.before_optimizations.txt"
    ));

    let args_desc = get_args_desc(input);

    println!("Kernel arguments: {:?}", args_desc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_type() -> Result<(), String> {
        assert_eq!(is_arg_type("f32"), true);
        Ok(())
    }

    #[test]
    fn test_get_type_shape_and_id() -> Result<(), String> {
        assert_eq!(
            get_type_shape_and_id("f32[500]{0} parameter(0)"),
            Some(("", KernelArgument::Typef32(500, 0)))
        );
        assert_eq!(
            get_type_shape_and_id("Arg_0_1 =  f32[500]{0} parameter(0)"),
            Some(("", KernelArgument::Typef32(500, 0)))
        );
        let input = r#"{
  ROOT fusion = f32[500]{0} fusion(Arg_0.1, Arg_1.2), kind=kLoop, calls=fused_computation, metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
"#;
        assert_eq!(get_type_shape_and_id(input), None);
        Ok(())
    }

    #[test]
    fn test_get_entry_block() -> Result<(), String> {
        let input: &str = r#"
  ROOT add.0 = f32[500]{0} add(broadcast.6, multiply.0), metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
}

ENTRY main.33 {
  Arg_0.1 = f32[500]{0} parameter(0), sharding={replicated}
  Arg_1.2 = f32[5]{0} parameter(1), sharding={replicated}
  ROOT fusion = f32[500]{0} fusion(Arg_0.1, Arg_1.2), kind=kLoop, calls=fused_computation, metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
}
something
"#;

        let expected_result = r#"{
  Arg_0.1 = f32[500]{0} parameter(0), sharding={replicated}
  Arg_1.2 = f32[5]{0} parameter(1), sharding={replicated}
  ROOT fusion = f32[500]{0} fusion(Arg_0.1, Arg_1.2), kind=kLoop, calls=fused_computation, metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
"#;

        assert_eq!(get_entry_block(input), expected_result);
        Ok(())
    }

    #[test]
    fn test_get_args_desc() -> Result<(), String> {
        let input: &str = r#"
  ROOT add.0 = f32[500]{0} add(broadcast.6, multiply.0), metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
}

ENTRY main.33 {
  Arg_0.1 = f32[500]{0} parameter(0), sharding={replicated}
  Arg_1.2 = f32[5]{0} parameter(1), sharding={replicated}
  ROOT fusion = f32[500]{0} fusion(Arg_0.1, Arg_1.2), kind=kLoop, calls=fused_computation, metadata={op_name="jit(_logsm_from_logmhalo_jax_kern)/jit(main)/add" source_file="/home/jacekczaja/JAX/gordon/gordon/sigmoid_smhm.py" source_line=137}
}
something
"#;

        let expected_result: Vec<KernelArgument> = vec![
            KernelArgument::Typef32(500, 0),
            KernelArgument::Typef32(5, 1),
        ];

        assert_eq!(get_args_desc(input), expected_result);
        Ok(())
    }

    #[test]
    fn test_parse_nonoptimized_hlo() -> Result<(), String> {
        let input = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/module_0002.jit__logsm_from_logmhalo_jax_kern.before_optimizations.txt"
        ));

        let expected_result: Vec<KernelArgument> = vec![
            KernelArgument::Typef32(5, 1),
            KernelArgument::Typef32(500, 0),
        ];

        assert_eq!(get_args_desc(input), expected_result);

        Ok(())
    }
    #[test]
    fn test_parse_optimized_hlo() -> Result<(), String> {
        let input = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/module_0002.jit__logsm_from_logmhalo_jax_kern.gpu_after_optimizations.txt"
        ));

        let expected_result: Vec<KernelArgument> = vec![
            KernelArgument::Typef32(500, 0),
            KernelArgument::Typef32(5, 1),
        ];

        assert_eq!(get_args_desc(input), expected_result);

        Ok(())
    }
}
