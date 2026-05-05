# ImageJ CLI テスト実行ガイド

このプロジェクトの ONNX プラグインをコマンドライン（CLI）からテスト・実行するための手順と注意点です。

## 1. 事前準備
テストを実行する前に、必ず最新のコードをビルドし、ImageJ のプラグインフォルダへコピーしてください。

```powershell
# ビルド
mvn clean package -DskipTests

# コピー（プロジェクトルートにあるバッチファイルを実行）
.\copy_to_plugins.bat
```

## 2. 実行コマンドの基本

### A. ヘッドレスモード（バッチ処理・自動テスト用）
GUIを表示せず、結果をコンソールに出力します。

```powershell
java "-Dplugins.dir=C:\tools\ImageJ" -Xmx2g -jar "C:\tools\ImageJ\ij.jar" -batch "test/test_all_models.ijm"
```

### B. GUIモード（マクロ実行確認用）
ImageJを起動し、指定したマクロを自動実行します。

```powershell
java "-Dplugins.dir=C:\tools\ImageJ" -Xmx2g -jar "C:\tools\ImageJ\ij.jar" -macro "test/test_all_models.ijm"
```

## 3. 重要な注意点

### `-Dplugins.dir` の指定（最重要）
`java -jar ij.jar` をプロジェクトディレクトリから実行すると、ImageJ はデフォルトで実行時のカレントディレクトリにある `plugins` フォルダを探してしまいます。
`C:\tools\ImageJ\plugins` にコピーした最新の JAR を読み込ませるためには、必ず **`-Dplugins.dir="C:\tools\ImageJ"`** を指定してください。

### パスの扱い（Windows）
- PowerShell では、`-D` オプション全体を `"` で囲む（例: `"-Dplugins.dir=..."`）ことで、引数の解釈エラーを防げます。
- マクロ内で指定するファイルパスは、バックスラッシュ（`\`）ではなくスラッシュ（`/`）を使用するか、ダブルバックスラッシュ（`\\`）を使用してください。

### マクロ引数とダイアログラベルの一致
`run("Command", "arg1=val1 arg2=val2")` で渡す引数名は、Java 側の `GenericDialog.add*Field("label", ...)` で指定したラベル名と**完全に一致**している必要があります（大文字小文字・アンダースコア等）。

### バッチモードでの ROI Manager / Results Table
- `-batch` モードでは、ROI Manager や Results Table のウィンドウは実体化されません。
- マクロ側で `roiManager("count")` 等の結果を取得したい場合は、Java 側で `rt.show("Results")` 等が適切に呼び出されている必要があります。
- ただし、ヘッドレス環境でのデッドロックを防ぐため、プラグイン側では `if (!IJ.isMacro()) rt.show("Results");` のように、マクロ実行時はウィンドウ表示を抑制する実装が推奨されます。

## 4. デバッグ方法
プラグイン内で `System.out.println("DEBUG: ...")` を使用すると、`-batch` モード実行時にコンソールへ直接ログが出力されます。実行が止まったり、意図した結果が出ない場合は、このログを確認してください。
