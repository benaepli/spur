use std::sync::Arc;
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
}

impl Backend {
    pub fn new(client: Client) -> Self {
        let documents: Arc<DashMap<Url, String>> = Arc::new(DashMap::new());
        let notify = Arc::new(Notify::new());

        // Spawn the debounced analysis loop.
        let analysis_client = client.clone();
        let analysis_docs = documents.clone();
        let analysis_notify = notify.clone();
        tokio::spawn(async move {
            analysis_loop(analysis_client, analysis_docs, analysis_notify).await;
        });

        Self {
            client,
            documents,
            notify,
        }
    }

    fn trigger_analysis(&self) {
        self.notify.notify_one();
    }
}

/// Long-lived task that debounces document changes and publishes diagnostics.
async fn analysis_loop(client: Client, documents: Arc<DashMap<Url, String>>, notify: Arc<Notify>) {
    const DEBOUNCE: Duration = Duration::from_millis(200);

    loop {
        // Wait for a change notification.
        notify.notified().await;

        // Debounce: keep resetting while new notifications arrive within the window.
        loop {
            tokio::select! {
                _ = notify.notified() => continue,
                _ = tokio::time::sleep(DEBOUNCE) => break,
            }
        }

        // Snapshot all documents and analyse each one.
        let snapshot: Vec<(Url, String)> = documents
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        for (uri, source) in snapshot {
            let src = source.clone();
            let result =
                tokio::task::spawn_blocking(move || spur_core::compiler::compile_lsp(&src)).await;

            let diagnostics = match result {
                Ok(compile_result) => {
                    let line_index = LineIndex::new(&source);
                    compile_result_to_diagnostics(&compile_result, &line_index)
                }
                Err(e) => {
                    // spawn_blocking panicked — report a single fallback diagnostic.
                    vec![Diagnostic {
                        range: Range::default(),
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("spur".into()),
                        message: format!("internal error: {e}"),
                        ..Default::default()
                    }]
                }
            };

            client.publish_diagnostics(uri, diagnostics, None).await;
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
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
