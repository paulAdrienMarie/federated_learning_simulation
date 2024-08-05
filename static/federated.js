import * as ort from "/dist/ort.training.wasm.min.js";

ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let NUMIMAGES = 15;
let BATCHSIZE = 10;

/**
 * Loads JSON from a given URL
 * @async
 * @load_Json
 * @param {String} path - Path to the json file to load
 * @returns {Promise<void>}
 */
async function loadJson(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const config = await response.json();
    return config;
  } catch (error) {
    console.error("Error loading config", error);
  }
}

/**
 * Runs the federated learning scenario for a 100 users
 * @async
 * @runFederated
 * @returns {Promise<Void>}
 */
export async function runFederated() {
  // Load the dataset to train on
  let dataset = await loadJson("/script/train.json");
  // Get the length of the dataset
  const datasetLength = Object.keys(dataset).length;
  // Initialize the number of users
  let numUsers = 100;
  // Initialize the number of completed users
  let completedUsers = 0;

  /**
   * Stop the current thread for a given time
   * @sleep
   * @param {Number} ms - Number of ms to wait
   * @returns {Promise<void>}
   */
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Runs the federated learning by batch of 20 users, 20 users run in parallel, in separate workers
   * @async
   * @param {Number} startIndex - Index to start the next batch of users
   * @returns {Promise<Void>}
   */
  async function runBatch(startIndex) {
    let promises = [];
    let workers = []; // Array to keep track of workers

    for (let j = 0; j < BATCHSIZE; j++) {
      let userIndex = startIndex + j;
      if (userIndex >= numUsers) break;
      console.log(`Creating Worker for user ${userIndex + 1}`);
      let i = (userIndex * NUMIMAGES) % datasetLength;
      let worker = new Worker("/script/worker.js", {
        type: "module",
      });

      // Send the data to the created Worker
      let data = {
        userId: userIndex + 1,
        dataset: Object.entries(dataset).slice(i, i + NUMIMAGES),
      };
      worker.postMessage(data);

      // Push the worker to the workers array
      workers.push(worker);

      promises.push(
        new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            console.log(`User ${e.data.userId} completed training.`);
            resolve();
          };

          worker.onerror = (e) => {
            console.error(`Error in worker for user ${userIndex + 1}:`, e);
            reject(e);
          };
        })
      );
    }
    sleep(2000);
    // Wait for all promises to resolve
    await Promise.all(promises);
    console.log("Terminating all workers");
    // Terminate all workers
    workers.forEach((worker) => worker.terminate());

    completedUsers += BATCHSIZE;
  }

  for (let i = 0; i < numUsers; i += BATCHSIZE) {
    await runBatch(i);
  }

  let completion_element = document.createElement("p");
  completion_element.id = "completion_element_id";
  completion_element.innerText = "Process terminated";
  let federated_button = document.getElementById("launch_federated");
  federated_button.appendChild(completion_element);
}
