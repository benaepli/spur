pub mod format;

use crate::parser::Span;
use phf_macros::phf_map;
use std::fmt;
use std::iter::Peekable;
use std::str::Chars;
use thiserror::Error;

/// A token represents a single meaningful unit in the source code with its position.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// The kind of token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,

    // Punctuation
    Comma,
    Semicolon,
    Colon,
    Dot,
    Underscore,
    At,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Question,
    QuestionQuestion,

    Equal,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Arrow,
    LeftArrow,
    ColonEqual,

    // Literals
    Identifier(String),
    String(String),
    Integer(i64),

    // Keywords
    ClientInterface,
    Role,
    Func,
    Var,
    Type,
    Return,
    For,
    Sync,
    Break,
    If,
    Else,
    In,
    And,
    Or,
    Map,
    List,
    Options,
    Append,
    Prepend,
    Min,
    Exists,
    Head,
    Tail,
    Len,
    RpcCall,
    True,
    False,
    Nil,
    Erase,
    Chan,
    Make,
    Send,
    Recv,
    Lock,
    CreateLock,
    Store,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Underscore => write!(f, "_"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::QuestionQuestion => write!(f, "??"),
            TokenKind::Equal => write!(f, "="),
            TokenKind::EqualEqual => write!(f, "=="),
            TokenKind::BangEqual => write!(f, "!="),
            TokenKind::Less => write!(f, "<"),
            TokenKind::LessEqual => write!(f, "<="),
            TokenKind::Greater => write!(f, ">"),
            TokenKind::GreaterEqual => write!(f, ">="),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::LeftArrow => write!(f, "<-"),
            TokenKind::ColonEqual => write!(f, ":="),
            TokenKind::Identifier(s) => write!(f, "{}", s),
            TokenKind::String(s) => write!(f, "\"{}\"", s),
            TokenKind::Integer(i) => write!(f, "{}", i),
            TokenKind::ClientInterface => write!(f, "ClientInterface"),
            TokenKind::Role => write!(f, "role"),
            TokenKind::Func => write!(f, "func"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Sync => write!(f, "sync"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::In => write!(f, "in"),
            TokenKind::And => write!(f, "and"),
            TokenKind::Or => write!(f, "or"),
            TokenKind::Map => write!(f, "map"),
            TokenKind::List => write!(f, "list"),
            TokenKind::Options => write!(f, "options"),
            TokenKind::Append => write!(f, "append"),
            TokenKind::Prepend => write!(f, "prepend"),
            TokenKind::Min => write!(f, "min"),
            TokenKind::Exists => write!(f, "exists"),
            TokenKind::Head => write!(f, "head"),
            TokenKind::Tail => write!(f, "tail"),
            TokenKind::Len => write!(f, "len"),
            TokenKind::RpcCall => write!(f, "rpc_call"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Nil => write!(f, "nil"),
            TokenKind::Erase => write!(f, "erase"),
            TokenKind::Chan => write!(f, "chan"),
            TokenKind::Make => write!(f, "make"),
            TokenKind::Send => write!(f, "send"),
            TokenKind::Recv => write!(f, "recv"),
            TokenKind::Lock => write!(f, "lock"),
            TokenKind::CreateLock => write!(f, "create_lock"),
            TokenKind::Store => write!(f, "store"),
        }
    }
}

static KEYWORDS: phf::Map<&'static str, TokenKind> = phf_map! {
    "ClientInterface" => TokenKind::ClientInterface,
    "role" => TokenKind::Role,
    "func" => TokenKind::Func,
    "var" => TokenKind::Var,
    "type" => TokenKind::Type,
    "return" => TokenKind::Return,
    "for" => TokenKind::For,
    "sync" => TokenKind::Sync,
    "break" => TokenKind::Break,
    "if" => TokenKind::If,
    "else" => TokenKind::Else,
    "in" => TokenKind::In,
    "and" => TokenKind::And,
    "or" => TokenKind::Or,
    "map" => TokenKind::Map,
    "list" => TokenKind::List,
    "options" => TokenKind::Options,
    "append" => TokenKind::Append,
    "prepend" => TokenKind::Prepend,
    "min" => TokenKind::Min,
    "exists" => TokenKind::Exists,
    "head" => TokenKind::Head,
    "tail" => TokenKind::Tail,
    "len" => TokenKind::Len,
    "rpc_call" => TokenKind::RpcCall,
    "true" => TokenKind::True,
    "false" => TokenKind::False,
    "nil" => TokenKind::Nil,
    "erase" => TokenKind::Erase,
    "chan" => TokenKind::Chan,
    "make" => TokenKind::Make,
    "send" => TokenKind::Send,
    "recv" => TokenKind::Recv,
    "lock" => TokenKind::Lock,
    "create_lock" => TokenKind::CreateLock,
    "store" => TokenKind::Store,
};

fn is_special_char(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')'
            | '{'
            | '}'
            | '['
            | ']'
            | ','
            | ';'
            | ':'
            | '.'
            | '+'
            | '-'
            | '*'
            | '/'
            | '%'
            | '!'
            | '?'
            | '>'
            | '<'
            | '='
            | '"'
            | '@'
    )
}

/// Errors that can occur during lexical analysis.
#[derive(Error, Debug, PartialEq, Clone)]
pub enum LexError {
    #[error("unexpected character")]
    UnexpectedChar(usize),
    #[error("unterminated string")]
    UnterminatedString(usize),
}

/// A lexical analyzer that converts source code into a stream of tokens.
pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    position: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given input string.
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.chars().peekable(),
            position: 0,
        }
    }

    /// Collects all tokens from the input, separating successful tokens from errors.
    pub fn collect_all(&mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        for result in self {
            match result {
                Ok(token) => tokens.push(token),
                Err(err) => errors.push(err),
            }
        }

        (tokens, errors)
    }

    fn match_next(&mut self, expected: char) -> bool {
        if self.input.peek() == Some(&expected) {
            self.input.next();
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.input.peek() {
            if ch.is_whitespace() {
                self.input.next();
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        // Skip the second '/'
        self.input.next();
        self.position += 1;

        while let Some(&ch) = self.input.peek() {
            self.input.next();
            self.position += 1;
            if ch == '\n' {
                break;
            }
        }
    }

    fn parse_string(&mut self, start: usize) -> Result<TokenKind, LexError> {
        let mut value = String::new();

        loop {
            let ch = self
                .input
                .next()
                .ok_or(LexError::UnterminatedString(start))?;
            self.position += 1;
            match ch {
                '"' => return Ok(TokenKind::String(value)),
                '\\' => {
                    // Handle escape sequences
                    match self.input.next() {
                        Some('"') => {
                            value.push('"');
                            self.position += 1;
                        }
                        Some('\\') => {
                            value.push('\\');
                            self.position += 1;
                        }
                        Some('n') => {
                            value.push('\n');
                            self.position += 1;
                        }
                        Some('t') => {
                            value.push('\t');
                            self.position += 1;
                        }
                        Some(ch) => {
                            // For other characters, include the backslash
                            value.push('\\');
                            value.push(ch);
                            self.position += 1;
                        }
                        None => {
                            return Err(LexError::UnterminatedString(start));
                        }
                    }
                }
                ch => {
                    value.push(ch);
                }
            }
        }
    }

    fn parse_number(&mut self, start: usize, first: char) -> Result<TokenKind, LexError> {
        let mut num_str = String::from(first);

        while let Some(&ch) = self.input.peek() {
            match ch {
                '0'..='9' => {
                    num_str.push(ch);
                    self.input.next();
                    self.position += 1;
                }
                _ => break,
            }
        }

        num_str
            .parse()
            .map(TokenKind::Integer)
            .map_err(|_| LexError::UnexpectedChar(start))
    }

    fn parse_identifier(&mut self, first: char) -> TokenKind {
        let mut result = String::from(first);
        while let Some(&ch) = self.input.peek()
            && !ch.is_whitespace()
            && !is_special_char(ch)
        {
            result.push(ch);
            self.input.next();
            self.position += 1;
        }
        KEYWORDS
            .get(&result as &str)
            .cloned()
            .unwrap_or_else(|| TokenKind::Identifier(result))
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_whitespace();

        let start = self.position;

        let ch = self.input.next()?;
        self.position += 1;

        let result = match ch {
            '(' => Ok(TokenKind::LeftParen),
            ')' => Ok(TokenKind::RightParen),
            '{' => Ok(TokenKind::LeftBrace),
            '}' => Ok(TokenKind::RightBrace),
            '[' => Ok(TokenKind::LeftBracket),
            ']' => Ok(TokenKind::RightBracket),
            ',' => Ok(TokenKind::Comma),
            ';' => Ok(TokenKind::Semicolon),
            ':' => {
                if self.match_next('=') {
                    Ok(TokenKind::ColonEqual)
                } else {
                    Ok(TokenKind::Colon)
                }
            }
            '.' => Ok(TokenKind::Dot),
            '@' => Ok(TokenKind::At),

            '_' => {
                // Check if this is part of an identifier or standalone underscore
                if let Some(&next_ch) = self.input.peek() {
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        // It's the start of an identifier like _test
                        Ok(self.parse_identifier('_'))
                    } else {
                        // Standalone underscore
                        Ok(TokenKind::Underscore)
                    }
                } else {
                    // End of input, standalone underscore
                    Ok(TokenKind::Underscore)
                }
            }

            '+' => Ok(TokenKind::Plus),
            '*' => Ok(TokenKind::Star),
            '%' => Ok(TokenKind::Percent),

            '-' => {
                if self.match_next('>') {
                    Ok(TokenKind::Arrow)
                } else {
                    Ok(TokenKind::Minus)
                }
            }

            '/' if self.input.peek() == Some(&'/') => {
                self.skip_line_comment();
                return self.next();
            }
            '/' => Ok(TokenKind::Slash),

            '!' => {
                if self.match_next('=') {
                    Ok(TokenKind::BangEqual)
                } else {
                    Ok(TokenKind::Bang)
                }
            }

            '?' => {
                if self.match_next('?') {
                    Ok(TokenKind::QuestionQuestion)
                } else {
                    Ok(TokenKind::Question)
                }
            }

            '=' => {
                if self.match_next('=') {
                    Ok(TokenKind::EqualEqual)
                } else {
                    Ok(TokenKind::Equal)
                }
            }

            '>' => {
                if self.match_next('=') {
                    Ok(TokenKind::GreaterEqual)
                } else {
                    Ok(TokenKind::Greater)
                }
            }

            '<' => {
                if self.match_next('-') {
                    Ok(TokenKind::LeftArrow)
                } else if self.match_next('=') {
                    Ok(TokenKind::LessEqual)
                } else {
                    Ok(TokenKind::Less)
                }
            }

            '"' => self.parse_string(start),

            '0'..='9' => self.parse_number(start, ch),

            ch if ch.is_alphabetic() || ch == '_' => Ok(self.parse_identifier(ch)),

            _ => Err(LexError::UnexpectedChar(start)),
        };

        let end = self.position;
        Some(result.map(|kind| {
            Token::new(
                kind,
                Span {
                    context: (),
                    start,
                    end,
                },
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let input = "func var if else for return";
        let mut lexer = Lexer::new(input);
        let (tokens, errors) = lexer.collect_all();

        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].kind, TokenKind::Func);
        assert_eq!(tokens[1].kind, TokenKind::Var);
        assert_eq!(tokens[2].kind, TokenKind::If);
        assert_eq!(tokens[3].kind, TokenKind::Else);
        assert_eq!(tokens[4].kind, TokenKind::For);
        assert_eq!(tokens[5].kind, TokenKind::Return);
    }

    #[test]
    fn test_operators() {
        let input = "+ - * / % == != < <= > >= -> =";
        let mut lexer = Lexer::new(input);
        let (tokens, errors) = lexer.collect_all();

        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 13);
        assert_eq!(tokens[0].kind, TokenKind::Plus);
        assert_eq!(tokens[1].kind, TokenKind::Minus);
        assert_eq!(tokens[5].kind, TokenKind::EqualEqual);
        assert_eq!(tokens[11].kind, TokenKind::Arrow);
    }

    #[test]
    fn test_identifiers_and_numbers() {
        let input = "myVar 123 _test ClientInterface";
        let mut lexer = Lexer::new(input);
        let (tokens, errors) = lexer.collect_all();

        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0].kind, TokenKind::Identifier(_)));
        assert_eq!(tokens[1].kind, TokenKind::Integer(123));
        assert!(matches!(tokens[2].kind, TokenKind::Identifier(_)));
        assert_eq!(tokens[3].kind, TokenKind::ClientInterface);
    }

    #[test]
    fn test_strings() {
        let input = r#""hello world" "escaped\"quote""#;
        let mut lexer = Lexer::new(input);
        let (tokens, errors) = lexer.collect_all();

        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].kind, TokenKind::String(_)));
    }

    #[test]
    fn test_comments() {
        let input = "func // this is a comment\nvar";
        let mut lexer = Lexer::new(input);
        let (tokens, errors) = lexer.collect_all();

        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].kind, TokenKind::Func);
        assert_eq!(tokens[1].kind, TokenKind::Var);
    }
}
