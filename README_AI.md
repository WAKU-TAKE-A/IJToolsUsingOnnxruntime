# エージェントのためのコード地図

この文書は、調査や変更の入口と、関連ファイルのつながりを示す。
操作方法・モデル仕様・制約は各 README、実際の振る舞いはソースコードを参照する。

## 説明を探す入口

| 調べたいこと | 参照先 |
|---|---|
| プロジェクト概要、ImageJ への導入 | [README.md](README.md) |
| RF-DETR の準備、対応モデル、ONNX 契約、検証コマンド | [RF-DETR/README.md](RF-DETR/README.md) |
| YOLO 各スクリプトの使い分け、保存先、既存の制約 | [yolo/README.md](yolo/README.md) |
| ImageJ CLI の起動方法 | [test/README_CLI_Testing.md](test/README_CLI_Testing.md) |
| Java の依存、コンパイル設定、JAR の構成 | [pom.xml](pom.xml) |
| 配布 JAR のライセンス同梱 | [LICENSE](LICENSE)、[NOTICE.txt](src/main/resources/META-INF/NOTICE.txt)、`pom.xml` の `resources` |

## 実行経路をたどる

Java ソースは [src/main/java/io/github/wakutakea/ijtoolsusingonnxruntime/](src/main/java/io/github/wakutakea/ijtoolsusingonnxruntime/) にある。
Python は事前のモデル準備と検証を担当し、ImageJ 内の推論経路には入らない。

**モデルの準備と読み込み**

- RF-DETR：[export_rfdetr.py](RF-DETR/export_rfdetr.py) が ONNX と契約メタデータを生成する。
  受け手は `OrtModel.validate()` → `ModelContract.validate()` → `RfDetrInference.validate()`。
- YOLO：[download_and_convert.py](yolo/download_and_convert.py) または [convert.py](yolo/convert.py) が ONNX を生成する。
  受け手は `ModelContract` の検証・クラス数推定、`OrtModel` のクラス名の解釈と、各タスクのデコーダ。
- ImageJ メニューの入口は [plugins.config](src/main/resources/plugins.config)。
  読み込み操作は `ORT_1st_ReadAndRelease`、読み込み済みモデルは `OrtModel`、共有スロットは `OrtUtil` が持つ。

**推論と表示**

- YOLO 系：`ORT_2nd_* .run()` → `OrtUtil` の前処理 → `OrtModel.runInference()`
  → 同じタスククラス内の解析・NMS 呼び出し・結果出力。
- RF-DETR：`ORT_2nd_Detection / Segmentation / Pose` の RF-DETR 分岐
  → `RfDetrResults.run()` → `RfDetrInference.predict()` → `OrtModel.runInference()`。
  その後 `RfDetrInference.applyNms()` → `Nms` → `RfDetrResults.render()` へ進む。

## 変更・不具合調査の入口

表中の Java ファイル名は、上記の Java ソースディレクトリを基準とする。

| 目的・症状 | 最初に読む場所 | 一緒に追う場所 |
|---|---|---|
| モデルの読み込み失敗、スロット管理 | `ORT_1st_ReadAndRelease` の `readModel()`、`OrtModel.load()` | `OrtUtil` のスロット操作、`ModelContract.validate()` |
| モデル形式やタスクを追加 | `ModelContract` に1行追加（宣言順＝形式インデックス、変更禁止） | 該当デコーダの分岐と `DecoderTest`。新メニューなら `plugins.config` |
| RF-DETR の export、メタデータ不整合 | `export_rfdetr.py` の `export_one()` | `RfDetrInference.validate()`、`ModelContract.resolveNumClasses()`。生成側と受け側を対で読む |
| YOLO の取得先、変換条件、ラベル抽出 | `yolo/` の該当スクリプト | [YOLO README](yolo/README.md) のスクリプト別の挙動と制約 |
| 推論への入力値・座標が違う | YOLO 系は `OrtUtil.preprocess()`、RF-DETR は `RfDetrInference.preprocess()` | 座標復元、各パーサー、`RfDetrInferenceTest` |
| ラベル・クラス数がおかしい | `ModelContract.resolveNumClasses()`、`OrtModel.loadClassNamesFromMetadata()` | RF-DETR は `export_one()` のスロット対応と `decode()`、YOLO は該当デコーダ。同梱 manifest（`ModelManifest`）がある場合は形状推定より manifest が優先される |
| 重複した枠や NMS 閾値の問題 | `Nms.suppress()` と呼び出し元 | `RfDetrResults.run()`、各 `ORT_2nd_*` の通常 / E2E 分岐、`NmsTest` |
| マスクやキーポイントの対応が崩れる | RF-DETR は `Detection.query`、`decode()`、`applyNms()` | `RfDetrResults.render()`。YOLO は該当タスククラス内の解析・出力 |
| テーブル・ROI・Overlay の問題 | RF-DETR は `RfDetrResults.render()`、YOLO は各 `output*()` | `OrtUtil` の結果管理、`enableRefreshData`、マクロ実行時の分岐 |
| ダイアログとマクロで挙動が違う | 該当クラスの `showDialog()`、`dialogItemChanged()` | ラベル名、`getNext*()` の順序、`test/test_rfdetr.ijm` |
| 検証環境や実行パスの問題 | `RF-DETR/verify_rfdetr.py` の `main()` | `export_rfdetr.py` の `ROOT / MODEL_DIR`、各 `setup_venv.bat`、`.gitignore` |

変更の波及を確認する境界は次の4箇所。

- **Python → ONNX → Java**：export の変更は、Java が読む形状・名前・メタデータに波及する。
- **検出候補 → NMS → 表示**：枠を選び直しても、マスクやキーポイントとの対応を保持する。
- **ダイアログ → マクロ**：フィールド名と取得順序の変更は、記録済みマクロに波及する。
- **通常形式 → E2E 形式**：同じタスク内にも別経路がある。片方の修正を他方へ機械的に広げない。

## 変更に対応する検証を探す

| 変更した場所 | 検証の入口 |
|---|---|
| 共通 NMS、検出データの保持 | [NmsTest.java](src/test/java/io/github/wakutakea/ijtoolsusingonnxruntime/NmsTest.java) |
| RF-DETR の前処理・解析・補間 | [RfDetrInferenceTest.java](src/test/java/io/github/wakutakea/ijtoolsusingonnxruntime/RfDetrInferenceTest.java) |
| 実モデル推論、描画、テーブル、NMS の接続 | [RfDetrSmoke.java](src/test/java/io/github/wakutakea/ijtoolsusingonnxruntime/RfDetrSmoke.java) |
| モデル読み込みと ImageJ コマンド・ROI Manager | [test_rfdetr.ijm](test/test_rfdetr.ijm) |
| Java 実行と公式 Python 処理の比較をまとめて実行 | [verify_rfdetr.py](RF-DETR/verify_rfdetr.py) |

`RfDetrSmoke` は `verify_rfdetr.py` から起動する実モデル用ハーネスで、JUnit の自動実行対象ではない。
Python 比較が見るのは NMS 前、追加 NMS の確認は Java ハーネスとマクロが担当する。
実行手順と必要なモデルは [RF-DETR README](RF-DETR/README.md) を参照する。

結果を追う場合は `RF-DETR/models/verification/` の `java.log`、`macro.log`、`report.json`、
各 variant の描画画像・CSV を見る。生成物は Git に含まれないため、新しい環境に存在するとは限らない。
YOLO の旧マクロを使う場合は [test_all_models.ijm](test/test_all_models.ijm) などの固定パス・形式指定を先に読む。

## 正本とローカル生成物の区別

- 編集する実装は `src/`、`RF-DETR/` と `yolo/` のスクリプト。生成済み ONNX を変更しても実装変更にはならない。
- モデルの由来・形状は `RF-DETR/models/<variant>/manifest.json`、取得画像の出典は `models/samples/sources.json` にある。
- `.venv/`、`models/`、`target/`、`RF-DETR/.research/` は環境または生成・調査資料。設計の正本として読まない。
- 配置先を調べるときは [copy_to_plugins.bat](copy_to_plugins.bat) と `verify_rfdetr.py` の `--macro` 経路を区別する。
  前者は通常の ImageJ インストール、後者は検証用の独立した配置先を更新する。

この地図を更新するのは、責務・入口・呼び出し関係・検証先が変わったとき。
操作手順や数値仕様の変更は、対応する README に記載する。
