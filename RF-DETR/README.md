# RF-DETR 用のモデル準備・検証

エージェント向けの全体構成と変更時の注意点は [README_AI.md](../README_AI.md) を参照してください。

このプロジェクトの推論本体は Java です。Python はモデルの取得・ONNX 変換と検証にだけ使います。
Python スクリプトと仮想環境はこの `RF-DETR/` フォルダに置き、重み、ONNX、サンプル画像、検証結果は
プロジェクト直下の `RF-DETR/models/` に保存します。

## セットアップと実行

プロジェクトルートで次を実行します（Python 3.12、Java 17 で検証）。

```powershell
.\RF-DETR\setup_venv.bat
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\export_rfdetr.py
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\verify_rfdetr.py
# ImageJ マクロ経由の読み込み・ROI Manager も検証する場合（AWT が使える環境）
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\verify_rfdetr.py --macro
```

既定では Detection Nano、Segmentation Nano、Keypoint Preview の3モデルを準備します。
重みと Python 依存ライブラリを合わせて数 GB の空き容量が必要です。
Python 側の `rfdetr` は 1.11.2、ONNX Runtime は Java と同じ 1.25.1 に固定しています。

`verify_rfdetr.py` は Zulu 17 と NetBeans 同梱 Maven を既定で使用します。
別環境では `--java-home`、`--maven`、`--imagej-jar` を指定できます。
このスクリプトは Maven のテスト・ビルドを実行し、Java で実推論・テーブル・ROI 描画・追記/更新を確認した後、
公式 Python の前処理・後処理と数値を比較します。GUI ウィンドウの操作テストは含みません。
出力先は `RF-DETR/models/verification/` です。
`--macro` は検証用の独立した plugins フォルダを作成するため、通常の ImageJ インストールには変更を加えません。

## 対応モデルとライセンス

2026-10-17 に公式資料と rfdetr 1.11.2 を確認しました。

| タスク | `--models` の指定 | Python クラス | 既定の入力解像度 | ライセンス |
|---|---|---|---|---|
| 物体検出 | `detection-nano` | `RFDETRNano` | 384 | Apache-2.0 |
| 物体検出 | `detection-small` | `RFDETRSmall` | 512 | Apache-2.0 |
| 物体検出 | `detection-medium` | `RFDETRMedium` | 576 | Apache-2.0 |
| 物体検出 | `detection-large` | `RFDETRLarge` | 704 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-nano` | `RFDETRSegNano` | 312 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-small` | `RFDETRSegSmall` | 384 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-medium` | `RFDETRSegMedium` | 432 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-large` | `RFDETRSegLarge` | 504 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-xlarge` | `RFDETRSegXLarge` | 624 | Apache-2.0 |
| インスタンスセグメンテーション | `segmentation-2xlarge` | `RFDETRSeg2XLarge` | 768 | Apache-2.0 |
| 人物の姿勢推定（17点、Preview） | `keypoint-preview` | `RFDETRKeypointPreview` | 576 | Apache-2.0 |

物体検出の XLarge / 2XLarge は **Platform Model License 1.0** で、Apache-2.0 ではありません。
今回の取得スクリプトには含めていません。分類・OBB 用の RF-DETR モデルは今回の公式モデル一覧にはありません。
Keypoint Preview は試験公開モデルであり、今後 API や重みが変わる可能性があります。
Java の共通出力処理はモデルのサイズに依存しませんが、実モデルでの確認対象は既定の3モデルです。

別サイズの取得例：

```powershell
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\export_rfdetr.py --models detection-small segmentation-medium
# Apache-2.0 の11モデルすべてを取得（大容量）
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\export_rfdetr.py --all
# 同じ構成で追加学習した重み
.\RF-DETR\.venv\Scripts\python.exe .\RF-DETR\export_rfdetr.py --models detection-nano --checkpoint C:\models\custom.pth
```

既存の同名 ONNX は再エクスポートされます。重みと画像はキャッシュを再利用します。
独自モデルでは学習データ・重みに付随するライセンスを別途確認してください。

## ImageJ からの使用

1. `mvn package` で作成した `target/IJTools_UsingOnnxruntime.jar` を ImageJ の `plugins` に配置します。
2. `Plugins > ORT > 1st Read and Release` で `RF-DETR/models/<variant>/model.onnx` を選びます。
3. `model_format` は次の表に合わせます。

| モデル | model_format | 実行メニュー |
|---|---|---|
| detection-* | `RF_DETR_Object` | `2nd Detection` |
| segmentation-* | `RF_DETR_Segment` | `2nd Segmentation` |
| keypoint-preview | `RF_DETR_Pose` | `2nd Pose` |

RF-DETR の出力には共通クラス `Nms` によるクラス別 NMS を追加適用します。
既存ダイアログの `nms_threshold` が有効です。同じクラスの枠の IoU が閾値を超えると、
信頼度の低い枠を除外します。`1.0` にすると NMS による除外を行いません。
異なるクラス（例：car と truck）の枠は相互に除外しません。
Pose は既存と同じく矩形 ROI とキーポイント ROI を ROI Manager に出力します。
Segmentation は輪郭 ROI、半透明 Overlay、二値の統合マスクを出力できます。
`enable_refresh_data` で結果を更新するか追記するかを選べます。
マクロ例は `test/test_rfdetr.ijm` を参照してください。

## ONNX の契約

- 入力：静的 `[1,3,H,W]` float32、RGB、直接リサイズ（Letterbox なし）。
- リサイズ：半画素中心の bilinear、antialias なし。0–1 に変換して ImageNet mean/std で正規化。
- `dets`：`[1,Q,4]`、正規化された `cx,cy,w,h`。
- `labels`：`[1,Q,C]`、sigmoid 前のクラス logits。
- `masks`：Seg のみ `[1,Q,Hm,Wm]`、画像全域の mask logits。
- `keypoints`：Pose のみ `[1,Q,クラス別にパディングした点数,D]`。座標、findability、精度行列情報。
- 全 query/class 組から global top-K を取り、スコア閾値で絞ります。
- ImageJ の実行メニューでは、その後に共通のクラス別 NMS を適用します。
  公式 Python との数値比較は NMS 前の出力で行い、追加 NMS は別途テストします。
  NMS 後の描画・テーブルは `verification/<variant>/nms-0.10/` に保存します。
- Pose は公式の uncertainty trace fusion をスコアに適用します。
- COCO 事前学習モデルの疎なカテゴリ ID（1–90）を維持し、90番の toothbrush を背景と誤認しません。

準備スクリプトがクラス名、背景スロット、top-K、キーポイントのスキーマ、trace alpha を
`ijtools.rfdetr.*` メタデータに埋め込みます。Java は読み込み時に形式を検証します。
**準備スクリプトを通していない一般の ONNX は、そのままでは読み込みません。**
異なるエクスポート形式を形状だけで推測して誤解析しないためです。

## サンプルの出典

- 犬などの画像：<https://media.roboflow.com/dog.jpg>（[公式の実行例](https://rfdetr.roboflow.com/latest/learn/run/detection/)）
- 人物：COCO val2017 `000000000785.jpg`。COCO 配布バケットから取得：
  <https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000785.jpg>
- 取得 URL と SHA-256 は `RF-DETR/models/samples/sources.json` に保存します。
- サンプル画像の権利はモデルの Apache-2.0 と別です。画像は Git 管理に含めません。

公式資料：

- [ソースとライセンス](https://github.com/roboflow/rf-detr/tree/1.11.2)
- [Roboflow ライセンス一覧](https://roboflow.com/licensing)
- [Detection](https://rfdetr.roboflow.com/latest/learn/run/detection/)
- [Segmentation](https://rfdetr.roboflow.com/latest/learn/run/segmentation/)
- [Keypoints](https://rfdetr.roboflow.com/latest/learn/run/keypoints/)
- [ONNX Export](https://rfdetr.roboflow.com/latest/exports/onnx/)
