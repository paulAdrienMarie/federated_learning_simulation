import * as ort from "/dist/ort.training.wasm.min.js";

// Set up wasm paths
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let trainingSession = null;
let images = null;
let target_tensor = null;
let numEpochs = 2;
let num_training = 0;

const train = "/artifacts/training_model.onnx";
const eval_ = "/artifacts/eval_model.onnx";
const optimizer = "/artifacts/optimizer_model.onnx";
const checkpoint = "/artifacts/checkpoint";

const createOptions = {
  checkpointState: checkpoint,
  trainModel: train,
  evalModel: eval_,
  optimizerModel: optimizer,
};

export async function loadTrainingSession(createOptions) {
  console.log("Trying to load Training Session");

  try {
    trainingSession = await ort.TrainingSession.create(createOptions);
    console.log("Training session loaded");
    return trainingSession;
  } catch (err) {
    console.error("Error loading the training session:", err);
    throw err;
  }
}

// async function runTrainingEpoch(images, epoch, target_tensor) {
//   const epochStartTime = Date.now();
//   const lossNodeName = trainingSession.handler.outputNames[0];

//   console.log(
//     `TRAINING | Epoch ${epoch + 1} / ${numEpochs} | Starting Training ... `
//   );

//   for (let image of images) {
//     // create input
//     const feeds = {
//       pixel_values: image,
//       target: target_tensor,
//     };

//     const results = await trainingSession.runTrainStep(feeds);

//     const loss = results[lossNodeName].data;
//     self.postMessage({
//       loss: loss,
//     });
//     console.log(`LOSS: ${loss}`);

//     await trainingSession.runOptimizerStep();
//     await trainingSession.lazyResetGrad();
//   }

//   const epochTime = Date.now() - epochStartTime;

//   self.postMessage({
//     epochMessage: `Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`,
//   });

//   console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
// }

// async function paramsToUint8Buffer(params) {
//   let buffer = new ArrayBuffer(params.length * 4);

//   let dataView = new DataView(buffer);

//   for (let i = 0; i < params.length; i++) {
//     dataView.setFloat32(i * 4, params[i], true);
//   }

//   let parameters = new Uint8Array(buffer);

//   return parameters;
// }

// // Worker code for message handling
// self.addEventListener("message", async (event) => {
//   let data = event.data;

//   if (!trainingSession) {
//     await loadTrainingSession();
//   }

//   const params_before = await trainingSession.getContiguousParameters(true);

//   images = data.images.map(
//     (img) => new ort.Tensor(img.type, img.data, img.dims)
//   );
//   target_tensor = new ort.Tensor(
//     data.target_tensor.type,
//     data.target_tensor.data,
//     data.target_tensor.dims
//   );

//   const startTrainingTime = Date.now();
//   console.log("Training started");

//   // Run training loop
//   for (let epoch = 0; epoch < numEpochs; epoch++) {
//     await runTrainingEpoch(images, epoch, target_tensor);
//   }

//   num_training++;

//   const trainingTime = Date.now() - startTrainingTime;

//   self.postMessage({
//     status: "Training completed",
//     trainingTime: trainingTime,
//   });

//   let params = await trainingSession.getContiguousParameters(true);

//   let parameters = await paramsToUint8Buffer(params.cpuData);

//   try {
//     await trainingSession.loadParametersBuffer(parameters, true);
//     console.log("Training Session parameters updated");
//   } catch (err) {
//     console.log("Error:", err);
//   }

//   if (num_training===10) {
//     fetch("/update_model", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ updated_weights: params }),
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         // console.log(data);
//         console.log("Model parameters updated");
//       })
//       .catch((error) => {
//         console.log("Error:", error);
//       });
//     await trainingSession.release();
//     i = 0;
//   }

//   self.postMessage({
//     epochMessage: "Model parameters updated",
//     reload: true,
//   });
// });

// self.onerror = function (error) {
//   console.error("Worker error:", error);
// };
