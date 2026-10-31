const API_BASE = "http://localhost:8000";
const form = document.querySelector("#prediction-form");
const errorBox = document.querySelector("#error-message");
const submitButton = document.querySelector("#submit-button");
const apiStatus = document.querySelector("#api-status");
const apiPill = document.querySelector(".api-pill");
const resultTitle = document.querySelector("#result-title");
const resultSummary = document.querySelector("#result-summary");
const probabilityValue = document.querySelector("#probability-value");
const probabilityMeter = document.querySelector("#probability-meter");
const resultSymbol = document.querySelector(".result-symbol");

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/`);
    if (!response.ok) throw new Error("API unavailable");
    apiStatus.textContent = "API connected";
    apiPill.classList.add("connected");
  } catch {
    apiStatus.textContent = "API offline";
    apiPill.classList.add("offline");
  }
}

function clearResult() {
  errorBox.hidden = true;
  resultTitle.textContent = "Your subscription prediction will appear here.";
  resultSummary.textContent = "Complete the customer and campaign details, then select “Predict”.";
  probabilityValue.textContent = "—";
  probabilityMeter.style.width = "0%";
  resultSymbol.textContent = "↗";
}

function randomiseInputs() {
  const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
  const ranges = {
    age: [18, 75], balance: [-500, 10000], day: [1, 31],
    duration: [30, 1800], campaign: [1, 10], pdays: [-1, 999], previous: [0, 8]
  };
  Object.entries(ranges).forEach(([name, [min, max]]) => {
    form.elements[name].value = randomInt(min, max);
  });
  ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"].forEach((name) => {
    const select = form.elements[name];
    select.selectedIndex = randomInt(0, select.options.length - 1);
  });
  clearResult();
}

function collectFeatures() {
  const features = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    features[element.name] = element.type === "number" ? Number(element.value) : element.value;
  }
  return features;
}

document.querySelector("#randomize-inputs").addEventListener("click", randomiseInputs);
document.querySelector("#reset-form").addEventListener("click", () => {
  form.reset();
  clearResult();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submitButton.disabled = true;
  submitButton.innerHTML = "Predicting… <span>↻</span>";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({data: collectFeatures()})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The API rejected the request.");

    const probability = Number(payload.probability);
    if (!Number.isFinite(probability)) throw new Error("The API response did not include a valid probability.");
    const pct = (probability * 100).toFixed(2);
    probabilityValue.textContent = `${pct}%`;
    probabilityMeter.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;

    if (Number(payload.prediction) === 1) {
      resultTitle.textContent = "Likely to subscribe.";
      resultSummary.textContent = `The model estimates a ${pct}% probability that this client will subscribe to a term deposit.`;
      resultSymbol.textContent = "✓";
    } else {
      resultTitle.textContent = "Unlikely to subscribe.";
      resultSummary.textContent = `The model estimates a ${pct}% probability that this client will subscribe to a term deposit.`;
      resultSymbol.textContent = "—";
    }
    document.querySelector("#result-panel").scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    errorBox.textContent = error instanceof TypeError
      ? "Can't reach FastAPI. Start the backend on port 8000."
      : (error.message || "Something went wrong. Please try again.");
    errorBox.hidden = false;
    await checkApi();
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = "Predict <span>↗</span>";
  }
});

checkApi();
