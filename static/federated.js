import * as ort from "/dist/ort.training.wasm.min.js";
import {
  predict,
  loadJSON,
  preprocessImageTraining,
  createTargetTensor,
} from "./App.js";
import { loadTrainingSession } from "./training-worker.js";

ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let NUMIMAGES = 7
let NUMEPOCHS = 2;

const ARTIFACTS_PATH = {
  checkpointState: "/artifacts/checkpoint",
  trainModel: "/artifacts/training_model.onnx",
  evalModel: "/artifacts/eval_model.onnx",
  optimizerModel: "/artifacts/optimizer_model.onnx",
};

async function runTrainingEpoch(images, epoch, target_tensor, session) {
  const epochStartTime = Date.now();
  const lossNodeName = session.handler.outputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${NUMEPOCHS} | Starting Training ... `
  );

  for (let image of images) {
    // create input
    const feeds = {
      pixel_values: image,
      target: target_tensor,
    };

    const results = await session.runTrainStep(feeds);

    const loss = results[lossNodeName].data;

    console.log(`LOSS: ${loss}`);

    await session.runOptimizerStep();
    await session.lazyResetGrad();
  }

  const epochTime = Date.now() - epochStartTime;

  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}

async function train(base64, new_classe, session) {
  let target_tensor = createTargetTensor(new_classe);

  let pre = await loadJSON("/script/preprocessor_config.json");

  let images = await preprocessImageTraining(base64, pre);

  images.map((img) => new ort.Tensor(img.type, img.data, img.dims));

  const startTrainingTime = Date.now();
  console.log("Training started");

  // Run training loop
  for (let epoch = 0; epoch < NUMEPOCHS; epoch++) {
    await runTrainingEpoch(images, epoch, target_tensor, session);
  }

  const trainingTime = Date.now() - startTrainingTime;

  console.log(`Training completed in ${trainingTime}`);
}

export async function runFederated() {
  // get the dataset from json file, dict{ key, class }
  let dataset = await loadJSON("/script/train.json");

  // get the base64 representation of images from json file, dict{ key, base64 }
  let base64data = await loadJSON("/script/train_base64images.json");

  // Initialize batch index
  let i = 0;
  // Initialize user index
  let j = 1;
  // Initialize the training session at null
  let trainingSession = null;
  // Iterate over the dataset in chunks of 7
  while (i < Object.keys(dataset).length) {
    // Get a slice of the entries
    const entriesSlice = Object.entries(dataset).slice(i, i + NUMIMAGES);
    console.log(`CURRENTLY RUNNING USER N ${j}`);
    // Loading training session
    console.log(`LOADING TRAINING SESSION FOR USER ${j}`);

    if (!trainingSession) {
      trainingSession = await loadTrainingSession(ARTIFACTS_PATH);
    }
    // Loop over the sliced entries
    for (const [key, value] of entriesSlice) {
      let label = await predict(base64data[key]);
      console.log(
        `Chat GPT predicted ${value}, ONNX model predicted ${
          Object.keys(label)[0]
        }`
      );

      if (value !== Object.keys(label)[0]) {
        await train(base64data[key], value, trainingSession); // retrain the model on the output of chatgpt
      }
    }
    // Update index for the next chunk
    i += 7;

    // Retrieve the params from the training session
    let params = await trainingSession.getContiguousParameters(true);

    // Make a request to the python server to store the new weights of the current user
    fetch("/update_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        updated_weights: params,
        user_id: j,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Model parameters updated");
      })
      .catch((error) => {
        console.log("Error:", error);
      });
    // Release the training session
    await trainingSession.release();

    // trainingSession back to null to have a new instance for the next user
    trainingSession = null;

    // update user index
    j++;
  }
}
