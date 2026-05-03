use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::Notify;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::convert::LineIndex;
use crate::diagnostics::compile_result_to_diagnostics;

/// Shared state for the language server.
pub struct Backend {
    client: Client,
    documents: Arc<DashMap<Url, String>>,
    notify: Arc<Notify>,
    /// Whether refinement checking should run on save / on debounce.
    /// Toggled via the `spur.refinements.onSave` LSP setting; ignored
    /// (always treated as `false`) when the binary was built without
    /// the `formulog` feature.
    refinements_on_save: Arc<AtomicBool>,
}

impl Backend {
    pub fn new(client: Client) -> Self {
        let documents: Arc<DashMap<Url, String>> = Arc::new(DashMap::new());
        let notify = Arc::new(Notify::new());
        let refinements_on_save = Arc::new(AtomicBool::new(false));

        let analysis_client = client.clone();
        let analysis_docs = documents.clone();
        let analysis_notify = notify.clone();
        let refinements_flag = refinements_on_save.clone();
        tokio::spawn(async move {
            analysis_loop(
                analysis_client,
                analysis_docs,
                analysis_notify,
                refinements_flag,
            )
            .await;
        });

        Self {
            client,
            documents,
            notify,
            refinements_on_save,
        }
    }

    fn trigger_analysis(&self) {
        self.notify.notify_one();
    }
}

/// Long-lived task that debounces document changes and publishes diagnostics.
async fn analysis_loop(
    client: Client,
    documents: Arc<DashMap<Url, String>>,
    notify: Arc<Notify>,
    refinements_on_save: Arc<AtomicBool>,
) {
    const DEBOUNCE: Duration = Duration::from_millis(200);

    loop {
        notify.notified().await;
        loop {
            tokio::select! {
                _ = notify.notified() => continue,
                _ = tokio::time::sleep(DEBOUNCE) => break,
            }
        }

        let snapshot: Vec<(Url, String)> = documents
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let want_refinements = refinements_on_save.load(Ordering::Relaxed);

        for (uri, source) in snapshot {
            let src = source.clone();
            let result = if want_refinements {
                tokio::task::spawn_blocking(move || analyze_with_refinements(&src)).await
            } else {
                tokio::task::spawn_blocking(move || spur_core::compiler::compile_lsp(&src)).await
            };

            let diagnostics = match result {
                Ok(compile_result) => {
                    let line_index = LineIndex::new(&source);
                    compile_result_to_diagnostics(&compile_result, &line_index)
                }
                Err(e) => vec![Diagnostic {
                    range: Range::default(),
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("spur".into()),
                    message: format!("internal error: {e}"),
                    ..Default::default()
                }],
            };

            client.publish_diagnostics(uri, diagnostics, None).await;
        }
    }
}

/// Run the full pipeline including the SMT-backed refinement check.
/// When `formulog` is disabled this is just a thin wrapper around
/// `compile_lsp`, so users who flip the setting on a non-formulog build
/// silently get the regular diagnostics.
#[cfg(feature = "formulog")]
fn analyze_with_refinements(source: &str) -> spur_core::compiler::CompileResult {
    let bin = match spur_liquid::flg_binary_path() {
        Some(p) => p,
        None => return spur_core::compiler::compile_lsp(source),
    };
    spur_core::compiler::compile_with_refinements(
        source,
        "<lsp>",
        &bin,
        Duration::from_secs(15),
    )
    .unwrap_or_else(|_| spur_core::compiler::compile_lsp(source))
}

#[cfg(not(feature = "formulog"))]
fn analyze_with_refinements(source: &str) -> spur_core::compiler::CompileResult {
    spur_core::compiler::compile_lsp(source)
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        // Honour `spur.refinements.onSave` if the editor supplied
        // initialization options.
        if let Some(opts) = params.initialization_options.as_ref() {
            let on_save = opts
                .get("spur")
                .and_then(|s| s.get("refinements"))
                .and_then(|r| r.get("onSave"))
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            self.refinements_on_save.store(on_save, Ordering::Relaxed);
        }
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "spur-lsp initialized")
            .await;
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        // Allow `workspace/didChangeConfiguration` to flip the
        // refinements-on-save setting without restarting the server.
        let on_save = params
            .settings
            .get("spur")
            .and_then(|s| s.get("refinements"))
            .and_then(|r| r.get("onSave"))
            .and_then(|b| b.as_bool());
        if let Some(b) = on_save {
            self.refinements_on_save.store(b, Ordering::Relaxed);
            self.trigger_analysis();
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.documents
            .insert(params.text_document.uri, params.text_document.text);
        self.trigger_analysis();
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.documents.insert(params.text_document.uri, change.text);
            self.trigger_analysis();
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);
        // Clear diagnostics for the closed document.
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}
