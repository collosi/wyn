use std::sync::OnceLock;

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use wyn_core::ast::NodeCounter;
use wyn_core::module_manager::{ModuleManager, PreElaboratedPrelude};

/// Cached prelude to avoid re-parsing for each document check
static PRELUDE_CACHE: OnceLock<PreElaboratedPrelude> = OnceLock::new();

fn get_prelude() -> &'static PreElaboratedPrelude {
    PRELUDE_CACHE.get_or_init(|| {
        ModuleManager::create_prelude().expect("Failed to create prelude cache")
    })
}

#[derive(Debug)]
struct Backend {
    client: Client,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self { client }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "wyn-analyzer".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "wyn-analyzer initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.on_change(TextDocumentItem {
            uri: params.text_document.uri,
            language_id: params.text_document.language_id,
            version: params.text_document.version,
            text: params.text_document.text,
        })
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().next() {
            self.on_change(TextDocumentItem {
                uri: params.text_document.uri,
                language_id: "wyn".to_string(),
                version: params.text_document.version,
                text: change.text,
            })
            .await;
        }
    }

    async fn did_close(&self, _: DidCloseTextDocumentParams) {
        // Clear diagnostics when file is closed
    }

    async fn hover(&self, _params: HoverParams) -> Result<Option<Hover>> {
        // TODO: Implement hover information
        Ok(None)
    }

    async fn completion(&self, _params: CompletionParams) -> Result<Option<CompletionResponse>> {
        // TODO: Implement completion
        Ok(None)
    }

    async fn goto_definition(
        &self,
        _params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        // TODO: Implement go-to-definition
        Ok(None)
    }
}

impl Backend {
    async fn on_change(&self, doc: TextDocumentItem) {
        let diagnostics = self.check_document(&doc.text);
        self.client
            .publish_diagnostics(doc.uri, diagnostics, Some(doc.version))
            .await;
    }

    fn check_document(&self, text: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Try to parse and type-check the document
        let result = wyn_core::Compiler::parse(text).and_then(|parsed| {
            let node_counter = NodeCounter::new();
            let module_manager = ModuleManager::from_prelude(get_prelude(), node_counter);
            parsed.elaborate(module_manager)?.resolve()?.type_check()
        });

        if let Err(e) = result {
            let range = if let Some(span) = e.span() {
                Range {
                    start: Position {
                        line: span.start_line.saturating_sub(1) as u32,
                        character: span.start_col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: span.end_line.saturating_sub(1) as u32,
                        character: span.end_col.saturating_sub(1) as u32,
                    },
                }
            } else {
                Range {
                    start: Position { line: 0, character: 0 },
                    end: Position { line: 0, character: 1 },
                }
            };

            diagnostics.push(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("wyn-analyzer".to_string()),
                message: e.to_string(),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        diagnostics
    }
}

#[tokio::main]
async fn main() {
    // Pre-initialize the prelude cache before starting the server
    // so any errors are caught early
    let _ = get_prelude();

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
