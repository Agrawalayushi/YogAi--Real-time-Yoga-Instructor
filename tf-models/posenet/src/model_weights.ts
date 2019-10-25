import * as tf from '@tensorflow/tfjs-core';

export class ModelWeights {
  private variables: {[varName: string]: tf.Tensor};

  constructor(variables: {[varName: string]: tf.Tensor}) {
    this.variables = variables;
  }

  weights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/weights`] as tf.Tensor4D;
  }

  depthwiseBias(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/biases`] as tf.Tensor1D;
  }

  convBias(layerName: string) {
    return this.depthwiseBias(layerName);
  }

  depthwiseWeights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/depthwise_weights`] as
        tf.Tensor4D;
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
