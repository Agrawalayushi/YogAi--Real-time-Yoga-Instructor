import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {BaseModel, PoseNetOutputStride} from './posenet_model';

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32') {
      input = input.toFloat();
    }
    const imageNetMean = tf.tensor([-123.15, -115.90, -103.06]);
    return input.add(imageNetMean);
  });
}

export class ResNet implements BaseModel {
  readonly model: tfconv.GraphModel;
  readonly outputStride: PoseNetOutputStride;

  constructor(model: tfconv.GraphModel, outputStride: PoseNetOutputStride) {
    this.model = model;
    const inputShape =
        this.model.inputs[0].shape as [number, number, number, number];
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be equal to or -1`);
    this.outputStride = outputStride;
  }

  predict(input: tf.Tensor3D): {[key: string]: tf.Tensor3D} {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [displacementFwd4d, displacementBwd4d, offsets4d, heatmaps4d] =
          this.model.predict(asBatch) as tf.Tensor[];

      const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze() as tf.Tensor3D;
      const displacementFwd = displacementFwd4d.squeeze() as tf.Tensor3D;
      const displacementBwd = displacementBwd4d.squeeze() as tf.Tensor3D;

      return {
        heatmapScores,
        offsets: offsets as tf.Tensor3D,
        displacementFwd: displacementFwd as tf.Tensor3D,
        displacementBwd: displacementBwd as tf.Tensor3D
      };
    });
  }

  dispose() {
    this.model.dispose();
  }
}
