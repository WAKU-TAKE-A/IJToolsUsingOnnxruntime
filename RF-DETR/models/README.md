# RF-DETR モデル保存先

`RF-DETR/export_rfdetr.py` が重み・ONNX・サンプル画像をここに保存します。
モデルと画像は Git 管理の対象外です。セットアップ・対応モデル・ライセンスは [RF-DETR/README.md](../README.md) を参照してください。

```text
RF-DETR/models/
  weights/                   元の事前学習済み重み
  detection-nano/model.onnx   ImageJ で使う検出モデル
  segmentation-nano/model.onnx
  keypoint-preview/model.onnx
  <variant>/manifest.json     ハッシュ、形状、バージョン、ラベル情報
  samples/                   サンプル画像と sources.json
  verification/              Java の描画画像・CSV・Python との比較レポート
  LICENSE-RF-DETR.txt         元モデルの Apache-2.0 ライセンス
```
