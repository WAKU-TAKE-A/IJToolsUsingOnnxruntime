## v0.9.5 の主な変更点

更新日：2026-10-17

### 新機能

- **Roboflow RF-DETR に対応**
  - 物体検出、インスタンスセグメンテーション、姿勢推定（Keypoint Preview）を既存の ImageJ メニューから実行できます。
  - モデル読み込み形式に `RF_DETR_Object`、`RF_DETR_Segment`、`RF_DETR_Pose` を追加しました。
- **共通 NMS を追加**
  - 重複検出の除去を共通クラス `Nms` に集約しました。
  - RF-DETR の3タスクでも `nms_threshold` が有効になり、同じクラスの重なった枠から低信頼度の枠を除去します。
  - 閾値を `1.0` にすると NMS による除外を行いません。
- **RF-DETR モデルの準備・検証ツール**
  - `RF-DETR/` に仮想環境セットアップ、モデル取得・ONNX 変換、Java と公式 Python の処理を比較するスクリプトを追加しました。
  - モデル、サンプル画像、検証結果は `RF-DETR/models/` に保存します。

### 修正・改善

- 配布 JAR に本プロジェクトと ONNX Runtime の MIT ライセンス全文を同梱し、第三者ライセンス通知も保持するようにしました。
- モデル読み込みダイアログの確定値を明示的に取得し、マクロで追加形式を選択できるようにしました。
- 推論入力テンソルとセッション設定のリソース解放を改善しました。
- YOLO の仮想環境セットアップが、バッチファイルの起動場所に依存しないよう修正しました。
- [RF-DETR の手順](RF-DETR/README.md)、[YOLO の手順](yolo/README.md)、エージェント向けの [コード地図](README_AI.md) を整備しました。

### 検証

- 共通 NMS と RF-DETR の前処理・後処理に対する16件のユニットテストを実施しました。
- Detection Nano、Segmentation Nano、Keypoint Preview の3モデルをサンプル画像で確認し、ImageJ マクロ実行と公式 Python の前処理・後処理との数値比較を行いました。
- `dog.jpg` の物体検出では、スコア閾値 `0.30`・NMS 閾値 `0.10` で person の重複が2件から1件になり、検出全体が7件から6件になります。

生成された `target/IJTools_UsingOnnxruntime.jar` を ImageJ の `plugins` に配置し、ImageJ を再起動してください。
RF-DETR の利用方法と対応範囲は [RF-DETR/README.md](RF-DETR/README.md) を参照してください。

---

## v0.9.4 の主な変更点

このバージョンでは、大幅な機能拡張として **インスタンスセグメンテーション（Seg）** および **回転矩形検出（OBB）** をサポートしました！

### 🚀 新機能
- **YOLO Segmentation 対応** (`2nd_Segmentation`)
  - 通常の `yolov8n-seg` 等の推論をサポートし、ImageJ の Polygon ROI として物体輪郭を出力します。
  - E2E（End-to-End）対応モデル (`yolo26n-seg` 等) にも対応しています。
- **YOLO OBB 対応** (`2nd_OBB`)
  - 航空写真などでよく使われる OBB（Oriented Bounding Box / 傾いた矩形）の推論をサポートしました (`yolov8s-obb` 等)。
  - E2E対応モデルの OBB 推論時、自動的に正しい外接矩形を計算・描画します。
- **README の更新**
  - 新機能を含めたプラグインの利用方法を分かりやすく更新しました。

### 🔧 修正・改善
- 推論時の NMS（非極大抑制）の際、OBB では正しく傾きを考慮した外接矩形で IoU を評価するようロジックを修正。
- バウンディングボックスの描画時に ImageJ の画面外にはみ出した際のクリッピング・座標ズレの問題を解消。

ぜひ新しい JAR ファイルを ImageJ の `plugins` フォルダに配置してお試しください。
