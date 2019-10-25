import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as mobilenet from './mobilenet';
import * as posenetModel from './posenet_model';
import * as resnet from './resnet';
import {toValidInputResolution} from './util';

describeWithFlags('PoseNet', NODE_ENVS, () => {
  let mobileNet: posenetModel.PoseNet;
  let resNet: posenetModel.PoseNet;
  const inputResolution = 513;
  const outputStride = 32;
  const multiplier = 1.0;
  const quantBytes = 4;
  const outputResolution = (inputResolution - 1) / outputStride + 1;
  const numKeypoints = 17;

  beforeAll((done) => {
    // Mock out the actual load so we don't make network requests in the unit
    // test.
    const resNetConfig =
        {architecture: 'ResNet50', outputStride, inputResolution, quantBytes} as
        posenetModel.ModelConfig;

    const mobileNetConfig = {
      architecture: 'MobileNetV1',
      outputStride,
      inputResolution,
      multiplier,
      quantBytes
    } as posenetModel.ModelConfig;

    spyOn(tfconv, 'loadGraphModel').and.callFake((): tfconv.GraphModel => {
      return null;
    });

    spyOn(resnet, 'ResNet').and.callFake(() => {
      return {
        outputStride,
        predict: (input: tf.Tensor3D) => {
          return {
            inputResolution,
            heatmapScores:
                tf.zeros([outputResolution, outputResolution, numKeypoints]),
            offsets: tf.zeros(
                [outputResolution, outputResolution, 2 * numKeypoints]),
            displacementFwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)]),
            displacementBwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)])
          };
        },
        dipose: () => {}
      };
    });

    spyOn(mobilenet, 'MobileNet').and.callFake(() => {
      return {
        outputStride,
        predict: (input: tf.Tensor3D) => {
          return {
            inputResolution,
            heatmapScores:
                tf.zeros([outputResolution, outputResolution, numKeypoints]),
            offsets: tf.zeros(
                [outputResolution, outputResolution, 2 * numKeypoints]),
            displacementFwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)]),
            displacementBwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)])
          };
        },
        dipose: () => {}
      };
    });

    posenetModel.load(resNetConfig)
        .then((posenetInstance: posenetModel.PoseNet) => {
          resNet = posenetInstance;
        })
        .then(() => posenetModel.load(mobileNetConfig))
        .then((posenetInstance: posenetModel.PoseNet) => {
          mobileNet = posenetInstance;
        })
        .then(done)
        .catch(done.fail);
  });


  it('estimateSinglePose does not leak memory', done => {
    const input =
        tf.zeros([inputResolution, inputResolution, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    resNet.estimateSinglePose(input, {flipHorizontal: false})
        .then(() => {
          return mobileNet.estimateSinglePose(input, {flipHorizontal: false});
        })
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimateMultiplePoses does not leak memory', done => {
    const input =
        tf.zeros([inputResolution, inputResolution, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;
    resNet
        .estimateMultiplePoses(input, {
          flipHorizontal: false,
          maxDetections: 5,
          scoreThreshold: 0.5,
          nmsRadius: 20
        })
        .then(() => {
          return mobileNet.estimateMultiplePoses(input, {
            flipHorizontal: false,
            maxDetections: 5,
            scoreThreshold: 0.5,
            nmsRadius: 20
          });
        })
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('load with mobilenet when input resolution is a number returns a model ' +
         'with a valid and the same input resolution width and height',
     (done) => {
       const inputResolution = 500;
       const validInputResolution =
           toValidInputResolution(inputResolution, outputStride);

       const expectedResolution = [validInputResolution, validInputResolution];

       posenetModel
           .load({architecture: 'MobileNetV1', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with resnet when input resolution is a number returns a model ' +
         'with a valid and the same input resolution width and height',
     (done) => {
       const inputResolution = 350;
       const validInputResolution =
           toValidInputResolution(inputResolution, outputStride);

       const expectedResolution = [validInputResolution, validInputResolution];

       posenetModel
           .load({architecture: 'ResNet50', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with mobilenet when input resolution is an object with width and height ' +
         'returns a model with a valid resolution for the width and height',
     (done) => {
       const inputResolution = {width: 600, height: 400};

       const expectedResolution = [
         toValidInputResolution(inputResolution.height, outputStride),
         toValidInputResolution(inputResolution.width, outputStride)
       ];

       posenetModel
           .load({architecture: 'MobileNetV1', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with resnet when input resolution is an object with width and height ' +
         'returns a model with a valid resolution for the width and height',
     (done) => {
       const inputResolution = {width: 700, height: 500};

       const expectedResolution = [
         toValidInputResolution(inputResolution.height, outputStride),
         toValidInputResolution(inputResolution.width, outputStride)
       ];

       posenetModel
           .load({architecture: 'ResNet50', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });
});
