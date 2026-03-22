import * as path from "path";
import {
  workspace,
  ExtensionContext,
} from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export function activate(context: ExtensionContext): void {
  const config = workspace.getConfiguration("spur");
  const serverPath =
    config.get<string>("serverPath") ||
    path.join(context.extensionPath, "server",
      process.platform === "win32" ? "spur-lsp.exe" : "spur-lsp");

  const serverOptions: ServerOptions = {
    command: serverPath,
    args: [],
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "spur" }],
  };

  client = new LanguageClient(
    "spur-lsp",
    "Spur Language Server",
    serverOptions,
    clientOptions
  );

  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
