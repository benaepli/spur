use std::fmt::Write;

pub struct BenchConfig {
    pub num_functions: usize,
    /// 1-5 for fixed complexity, 0 for round-robin mixed.
    pub complexity: u8,
}

pub fn generate_source(config: &BenchConfig) -> String {
    let mut out = String::new();

    for i in 0..config.num_functions {
        let level = if config.complexity == 0 {
            (i % 5) as u8 + 1
        } else {
            config.complexity
        };
        generate_function(&mut out, i, level);
        out.push('\n');
    }

    out.push_str(BOILERPLATE);
    out
}

fn generate_function(out: &mut String, index: usize, complexity: u8) {
    match complexity {
        1 => gen_level1(out, index),
        2 => gen_level2(out, index),
        3 => gen_level3(out, index),
        4 => gen_level4(out, index),
        _ => gen_level5(out, index),
    }
}

/// Identity: refined param, return same.
fn gen_level1(out: &mut String, i: usize) {
    writeln!(
        out,
        "fn bench_fn_{i}(p0: int {{ v | v > 0 }}): int {{ v | v > 0 }} {{\n    return p0;\n}}"
    )
    .unwrap();
}

/// Arithmetic: two positive params, return with tighter bound.
fn gen_level2(out: &mut String, i: usize) {
    writeln!(
        out,
        "fn bench_fn_{i}(p0: int {{ v | v > 0 }}, p1: int {{ v | v > 0 }}): int {{ v | v > 1 }} {{\n    \
         var r: int {{ v | v > 1 }} = p0 + p1;\n    return r;\n}}"
    )
    .unwrap();
}

/// Conditional: if-else with env refinement in each branch.
fn gen_level3(out: &mut String, i: usize) {
    writeln!(
        out,
        "fn bench_fn_{i}(p0: int {{ v | v >= 0 }}): int {{ v | v > 0 }} {{\n    \
         if (p0 > 0) {{\n        return p0;\n    }}\n    return 1;\n}}"
    )
    .unwrap();
}

/// Multi-param chain: three params with intermediate lets.
fn gen_level4(out: &mut String, i: usize) {
    writeln!(
        out,
        "fn bench_fn_{i}(p0: int {{ v | v > 0 }}, p1: int {{ v | v > 0 }}, p2: int {{ v | v > 0 }}): int {{ v | v > 2 }} {{\n    \
         var s1: int {{ v | v > 1 }} = p0 + p1;\n    \
         var s2: int {{ v | v > 2 }} = s1 + p2;\n    \
         return s2;\n}}"
    )
    .unwrap();
}

/// Deep chain: progressively tighter refinements through let-bindings.
fn gen_level5(out: &mut String, i: usize) {
    let base = 10;
    writeln!(
        out,
        "fn bench_fn_{i}(p0: int {{ v | v > {base} }}): int {{ v | v > {} }} {{\n    \
         var a: int {{ v | v > {} }} = p0 + 1;\n    \
         var b: int {{ v | v > {} }} = a + 1;\n    \
         var c: int {{ v | v > {} }} = b + 1;\n    \
         var d: int {{ v | v > {} }} = c + 1;\n    \
         return d;\n}}",
        base + 4,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
    )
    .unwrap();
}

const BOILERPLATE: &str = r#"
role Node {
    fn Init(me: int) {}
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) {}
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
