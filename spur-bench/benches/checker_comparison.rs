#[cfg(not(all(feature = "native-checker", feature = "formulog")))]
compile_error!("Both `native-checker` and `formulog` features must be enabled for this benchmark");

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spur_ast::NameId;
use spur_bench::{BenchConfig, generate_source};
use spur_core::compiler;
use spur_liquid::{check_native, check_with_formulog, clear_result_cache, flg_binary_path, lower_program};

fn flg_bin() -> std::path::PathBuf {
    flg_binary_path().expect("SPUR_FLG_BIN must be set or the binary must be built with formulog feature")
}

fn bench_end_to_end(c: &mut Criterion) {
    let flg = flg_bin();
    let sizes: &[usize] = &[10, 50, 100, 500];

    let mut group = c.benchmark_group("end_to_end");
    for &n in sizes {
        let source = generate_source(&BenchConfig {
            num_functions: n,
            complexity: 0,
        });

        if n >= 500 {
            group.sample_size(10);
        }

        group.bench_with_input(BenchmarkId::new("native", n), &source, |b, src| {
            b.iter(|| {
                compiler::compile_with_native_refinements(src, "bench").unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("formulog", n), &source, |b, src| {
            b.iter(|| {
                clear_result_cache();
                compiler::compile_with_refinements(src, "bench", &flg, Duration::from_secs(300))
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_checking_only(c: &mut Criterion) {
    let flg = flg_bin();
    let sizes: &[usize] = &[10, 50, 100, 500];

    let mut group = c.benchmark_group("checking_only");
    for &n in sizes {
        let source = generate_source(&BenchConfig {
            num_functions: n,
            complexity: 0,
        });

        let compile_result = compiler::compile(&source, "bench");
        assert!(
            compile_result.pure.is_some(),
            "compilation failed for size {n}"
        );
        let pure = compile_result.pure.unwrap();
        let liquid_out = lower_program(pure);
        let cprog = liquid_out.program;
        let fns: Vec<NameId> = cprog.funcs.iter().map(|f| f.name).collect();

        if n >= 500 {
            group.sample_size(10);
        }

        group.bench_with_input(BenchmarkId::new("native", n), &n, |b, _| {
            b.iter(|| check_native(&cprog, &fns).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("formulog", n), &n, |b, _| {
            b.iter(|| {
                clear_result_cache();
                check_with_formulog(&cprog, &fns, &flg, Duration::from_secs(300)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_complexity_scaling(c: &mut Criterion) {
    let flg = flg_bin();
    let n = 50;
    let complexities: &[u8] = &[1, 2, 3, 4, 5];

    let mut group = c.benchmark_group("complexity_scaling");
    for &complexity in complexities {
        let source = generate_source(&BenchConfig {
            num_functions: n,
            complexity,
        });

        let compile_result = compiler::compile(&source, "bench");
        let pure = compile_result.pure.unwrap();
        let liquid_out = lower_program(pure);
        let cprog = liquid_out.program;
        let fns: Vec<NameId> = cprog.funcs.iter().map(|f| f.name).collect();

        group.bench_with_input(
            BenchmarkId::new("native", complexity),
            &complexity,
            |b, _| {
                b.iter(|| check_native(&cprog, &fns).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("formulog", complexity),
            &complexity,
            |b, _| {
                b.iter(|| {
                    clear_result_cache();
                    check_with_formulog(&cprog, &fns, &flg, Duration::from_secs(300)).unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_end_to_end, bench_checking_only, bench_complexity_scaling);
criterion_main!(benches);
