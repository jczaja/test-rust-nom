// TODO: PArsing, find entry block "ENTRY" and then parameters scan for "parameter(" so they are valid
// within this block
// At the end I should now parameters per kernel
//
// sharding={replicated??} check if I can gen info on sharded configuration
// type od data and shape

use nom::{
    bytes::complete::is_a, bytes::complete::tag, bytes::complete::take_till,
    bytes::complete::take_until, character::complete::digit1, character::is_digit, error::Error,
    sequence::delimited, sequence::tuple, IResult,
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

fn get_shape(input: &str) -> (&str, u32) {
    let mut parse_shape = delimited(
        tag::<&str, &str, Error<&str>>("["),
        take_till(is_not_char_digit),
        tag("]"),
    );
    let (s, shape_str) = parse_shape(input).expect("Error: parsing shape failed!");
    let shape = str::parse::<u32>(shape_str).expect("Error: converting shape to int failed");
    (s, shape)
}

fn get_type_shape_and_id(input: &str) -> KernelArgument {
    let (s, argtype) = tag::<&str, &str, Error<_>>("f32")(input).expect("Error parsing type");
    let (s, shape) = get_shape(s);

    let (s, _) = take_until::<&str, &str, Error<&str>>("parameter")(s)
        .expect("Error: No parameter in input!!");
    let mut parse_id = delimited(
        tag::<&str, &str, Error<&str>>("parameter("),
        take_till(is_not_char_digit),
        tag(")"),
    );
    let (s, k) = parse_id(s).expect("Error: parsing arg Id failed!");

    let arg_id = str::parse::<u32>(k).expect("Error: converting to int failed");

    //TODO: match argtype
    // match argtype

    KernelArgument::Typef32(shape, arg_id)
}

fn get_args_desc(input: &str) -> Vec<KernelArgument> {
    let mut arguments: Vec<KernelArgument> = vec![];

    let (s, _) =
        take_until::<&str, &str, Error<&str>>("ENTRY")(input).expect("Error: No ENTRY in input!!");

    // in a loop execute till end

    //  let (s,_) = take_until::<&str, &str, Error<&str>>("parameter")(input).expect("Error: No parameter in input!!");
    let (s, _) = take_until::<&str, &str, Error<&str>>("parameter")(input)
        .expect("Error: No parameter in input!!");
    let mut parse_id = delimited(
        tag::<&str, &str, Error<&str>>("parameter("),
        take_till(is_not_char_digit),
        tag("),"),
    );
    let (s, k) = parse_id(s).expect("Error: parsing arg Id failed!");

    let arg_id = str::parse::<u8>(k).expect("Error: convertin to int failed");
    println!("ARG_ID: {arg_id}");

    println!("Found: {k}");
    println!("remainder: {s}");
    arguments
}

fn main() {
    println!("Hello Rust Parsers world!");
    assert_eq!(parser("Kupa Hello, World!"), Ok((", World!", "Hello")));

    let input = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/module_0002.jit__logsm_from_logmhalo_jax_kern.before_optimizations.txt"
    ));

    get_args_desc(input);

    //    assert!();
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
            KernelArgument::Typef32(500, 0)
        );
        Ok(())
    }

    #[test]
    fn test_get_shape() -> Result<(), String> {
        assert_eq!(get_shape("[500]"), ("", 500));
        assert_eq!(get_shape("[44]{0}"), ("{0}", 44));
        Ok(())
    }
}
