# YOLO モデル準備スクリプト

Java / ImageJ プラグイン用の ONNX を準備・検査する補助ツールです。
エージェント向けの全体説明は [README_AI.md](../README_AI.md) を参照してください。
RF-DETR 用の処理と仮想環境は [RF-DETR](../RF-DETR/README.md) に分離しています。

## セットアップと通常の取得

以下はプロジェクトルートから実行します。Python 3.12 のローカル環境があります。

```powershell
.\yolo\setup_venv.bat
.\yolo\.venv\Scripts\python.exe .\yolo\download_and_convert.py yolov8n-seg
.\yolo\.venv\Scripts\python.exe .\yolo\inspect_onnx.py .\yolo\models\yolov8n-seg.onnx
```

`setup_venv.bat` はスクリプト自身のフォルダを基準に `.venv` を作成し、pip 更新と依存のインストールを行います。
対話用の `pause` があるため、完全自動実行には次のように Python を直接指定します。

```powershell
python -m venv .\yolo\.venv
.\yolo\.venv\Scripts\python.exe -m pip install -r .\yolo\requirements.txt
```

依存は RF-DETR 側と同じく完全固定（`==`）です：`ultralytics==8.4.208`、`onnx==1.23.2`、
`onnxruntime==1.25.1`（Java 側の ONNX Runtime と一致）、`numpy==2.5.4`。
export 仕様を変えるときは、固定版数とこの文書を一緒に更新してください。

## スクリプト別の挙動

| スクリプト | 入力と用途 | 出力・書き込み |
|---|---|---|
| `download_and_convert.py` | モデル名を1引数。省略時 `yolov8n-seg`。`--opset`、`--end2end` 指定可 | スクリプト隣の `models/` に `.pt`、`.onnx`、クラス名 `.txt`、`<モデル>.onnx.manifest.json` |
| `convert.py` | `<model_path> [classes_file]`、旧変換とクラス絞り込み | Ultralytics が返す場所に ONNX。絞り込み時はその ONNX を上書き |
| `inspect_onnx.py` | ONNX の入力・出力名、形状、要素型の確認 | コンソールのみ。モデルは変更しない |
| `verify_yolo.py` | Java ハーネス（YoloSmoke）と Python リファレンスの NMS 前比較（検出の標準形式のみ） | `models/verification/` に `java.log`、`report.json`、ハーネス出力 |
| `extract_names.py` | ONNX の `names` メタデータを抽出 | ONNX と同じ場所・同じ stem の `.txt` を UTF-8 で作成・上書き |

`download_and_convert.py` は実行時に `yolo/models` へカレントディレクトリを変更します。
モデル名に `.pt` がなければ付加し、basename にした後 `YOLO()` に渡します。
したがって `C:\other\custom.pt` をそのままロードする用途には使えません。
export は opset（既定 17、RF-DETR 側の export と一致）と E2E の有無（既定は通常モデル）を
明示して渡します。既定値に依存するのはそれ以外の export 引数だけです。
export 後、ONNX の隣に `<モデル>.onnx.manifest.json`（RF-DETR 側と共通の形式：
由来・変換ツールと版数・opset・入出力形状・SHA-256・埋め込みメタデータ・クラス数）を書きます。
Java 側は同梱 manifest がある場合、その task とクラス数を形状推定より優先します。

`extract_names.py` の `.txt` は1行1クラスであり、辞書の ID 自体は書き出しません。
メタデータのキーは整数IDを想定しています。
ファイル不足やメタデータ不足を表示して終了する処理もあるため、成功時は実際の出力を確認します。

## 旧 convert.py を使う場合

```powershell
# 保存先を散らさないよう、モデル用の作業場所から実行する。
Push-Location .\yolo\models
try {
    ..\.venv\Scripts\python.exe ..\convert.py yolov8n.pt
} finally {
    Pop-Location
}
```

`convert.py` は opset 12 を指定します。引数やカレントディレクトリによって保存先が決まるため、
実際に表示された export パスを確認してください。
ローカルの追加学習済み `.pt` を使う場合も、対応するアーキテクチャとライセンスを確認します。

`classes_file` は保持するクラス ID を1行ずつ記載するファイルです（クラス名の一覧ではありません）。
このオプションには以下の制約があります。

- 第1出力が `[batch,4+classes,anchors]` の通常物体検出であることを前提に、軸1へ Gather を追加します。
- Pose、Seg、OBB、E2E、転置された出力を一般的に扱う機能ではありません。
- 絞り込み後のクラス ID は入力一覧の順に 0 から振り直します。
- 元の export 済み ONNX を上書きします。保持したい ONNX と別の作業先で実行してください。
- ID の重複・範囲検証がなく、数字でない行は無視し、指定ファイルが存在しない場合も明示的に失敗しません。
- 出力の batch / anchor 軸を動的にするので、Java 側のクラス数・レイアウト推定との適合確認が必要です。

## ImageJ 側の形式指定

ONNX の出力を確認して `1st Read and Release` の `model_format` を指定します。
通常の検出なら `YOLO_Object_Pixel`、E2E検出なら `YOLO_Object_E2E`、
分類は `YOLO_Class`、姿勢推定は `YOLO_Pose` / `YOLO_Pose_E2E`、
Seg は `YOLO_Segment` / `YOLO_Segment_E2E`、OBB は `YOLO_OBB` / `YOLO_OBB_E2E` です。
モデル名だけで通常形式と E2E を断定しないでください。

ラベルの正本は Java の `ModelContract`（宣言順＝形式インデックス、変更禁止）。
クラス名は通常 ONNX メタデータから読み込み、`.txt` があれば自動で読み込む実装ではありません。
NMS とテストの現状については [README_AI.md](../README_AI.md) を参照してください。
