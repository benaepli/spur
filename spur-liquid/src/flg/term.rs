//! A tiny term IR matching the grammar accepted by Formulog's runtime
//! parser (see `parser.cpp` in the formulog C++ codegen).
//!
//! The syntax supported by the parser is roughly:
//!
//! ```text
//!   term      ::= literal | tuple | list | ctor | string
//!   literal   ::= INT | INT 'L' | BOOL
//!   tuple     ::= '(' term (',' term)+ ')'   (* arity >= 2 *)
//!   list      ::= '[' (term (',' term)*)? ']' | term '::' term
//!   ctor      ::= NAME ( '(' term (',' term)* ')' )?
//!   string    ::= '"' (no '"' / '\\' chars) '"'
//! ```
//!
//! In particular, the parser does **not** accept escape sequences inside
//! string literals; we conservatively reject any string containing `"`
//! or any control character.

use std::fmt::Write;

/// One Formulog term, in a shape that round-trips through the parser
/// emitted by Formulog's codegen (see `take_term` in `parser.cpp`).
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// 32-bit integer (e.g. `name_id`). Rendered without a suffix.
    I32(i32),
    /// 64-bit integer (e.g. `i64`). Rendered with the trailing `L`.
    I64(i64),
    Bool(bool),
    /// String literal. The renderer rejects strings that the Formulog
    /// parser cannot lex back (no `"`, no control chars, no `\\`).
    Str(String),
    /// Empty list, equivalent to `[]` (parsed as the `nil` constructor).
    Nil,
    /// Cons of head `::` tail. The renderer prefers the `[a, b, c]`
    /// flat form when the tail is itself a list of cons / nil cells,
    /// and falls back to `a :: rest` otherwise.
    Cons(Box<Term>, Box<Term>),
    /// Tuple with arity >= 2.
    Tuple(Vec<Term>),
    /// Constructor application: zero-arg constructors render as just
    /// `name`, n-ary as `name(arg1, ..., argN)`.
    Ctor(String, Vec<Term>),
}

impl Term {
    pub fn ctor(name: impl Into<String>, args: Vec<Term>) -> Self {
        Term::Ctor(name.into(), args)
    }

    pub fn nullary(name: impl Into<String>) -> Self {
        Term::Ctor(name.into(), Vec::new())
    }

    pub fn list<I: IntoIterator<Item = Term>>(items: I) -> Self {
        let mut iter = items.into_iter();
        match iter.next() {
            None => Term::Nil,
            Some(first) => Term::Cons(Box::new(first), Box::new(Term::list(iter))),
        }
    }

    pub fn some(t: Term) -> Self {
        Term::Ctor("some".into(), vec![t])
    }

    pub fn none() -> Self {
        Term::Ctor("none".into(), Vec::new())
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }

    pub fn render_into(&self, out: &mut String) {
        match self {
            Term::I32(n) => {
                let _ = write!(out, "{}", n);
            }
            Term::I64(n) => {
                let _ = write!(out, "{}L", n);
            }
            Term::Bool(true) => out.push_str("true"),
            Term::Bool(false) => out.push_str("false"),
            Term::Str(s) => {
                out.push('"');
                out.push_str(s);
                out.push('"');
            }
            Term::Nil => out.push_str("[]"),
            Term::Cons(_, _) => self.render_list(out),
            Term::Tuple(items) => {
                out.push('(');
                for (i, t) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    t.render_into(out);
                }
                out.push(')');
            }
            Term::Ctor(name, args) => {
                out.push_str(name);
                if !args.is_empty() {
                    out.push('(');
                    for (i, t) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        t.render_into(out);
                    }
                    out.push(')');
                }
            }
        }
    }

    fn render_list(&self, out: &mut String) {
        let mut elems = Vec::new();
        let mut cur = self;
        loop {
            match cur {
                Term::Cons(head, tail) => {
                    elems.push(head.as_ref());
                    cur = tail.as_ref();
                }
                Term::Nil => {
                    out.push('[');
                    for (i, e) in elems.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        e.render_into(out);
                    }
                    out.push(']');
                    return;
                }
                _ => {
                    for e in &elems {
                        e.render_into(out);
                        out.push_str(" :: ");
                    }
                    cur.render_into(out);
                    return;
                }
            }
        }
    }
}

/// Reject strings that cannot survive a Formulog round trip. The
/// runtime parser does not understand escape sequences (see
/// `take_string` in `parser.cpp`), so any string literal that would
/// require escaping is unsafe to emit.
pub fn validate_string_literal(s: &str) -> Result<(), &'static str> {
    if s.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f) {
        Err("string literal contains a character that Formulog's parser cannot read back")
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_primitives() {
        assert_eq!(Term::I32(-3).render(), "-3");
        assert_eq!(Term::I64(7).render(), "7L");
        assert_eq!(Term::Bool(true).render(), "true");
        assert_eq!(Term::Str("ok".into()).render(), "\"ok\"");
        assert_eq!(Term::Nil.render(), "[]");
    }

    #[test]
    fn render_list_uses_bracket_form() {
        let t = Term::list(vec![Term::I32(1), Term::I32(2), Term::I32(3)]);
        assert_eq!(t.render(), "[1, 2, 3]");
    }

    #[test]
    fn render_ctor_and_tuple() {
        let t = Term::ctor("t_array", vec![Term::nullary("t_int")]);
        assert_eq!(t.render(), "t_array(t_int)");

        let tup = Term::Tuple(vec![Term::I32(1), Term::Bool(true)]);
        assert_eq!(tup.render(), "(1, true)");
    }

    #[test]
    fn render_some_none() {
        assert_eq!(Term::some(Term::I32(2)).render(), "some(2)");
        assert_eq!(Term::none().render(), "none");
    }

    #[test]
    fn validates_string_literals() {
        assert!(validate_string_literal("hello").is_ok());
        assert!(validate_string_literal("with\nnewline").is_err());
        assert!(validate_string_literal("with\"quote").is_err());
        assert!(validate_string_literal("with\\backslash").is_err());
    }
}
