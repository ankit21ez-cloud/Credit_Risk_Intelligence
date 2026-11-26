
import { useState } from "react";
import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [search, setSearch] = useState("");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [predictionResults, setPredictionResults] = useState([]);
  const [predictionSummary, setPredictionSummary] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  const applicants = [
    { id: "100001", name: "Applicant 100001", income: "₹45,000", credit: "₹2,50,000", risk: "Low Risk" },
    { id: "100002", name: "Applicant 100002", income: "₹28,000", credit: "₹4,00,000", risk: "High Risk" },
    { id: "100003", name: "Applicant 100003", income: "₹65,000", credit: "₹3,20,000", risk: "Low Risk" },
    { id: "100004", name: "Applicant 100004", income: "₹35,000", credit: "₹3,80,000", risk: "Medium Risk" },
    { id: "100005", name: "Applicant 100005", income: "₹22,000", credit: "₹5,00,000", risk: "High Risk" },
  ];

  const navigation = [
    { name: "Dashboard", icon: "▦" },
    { name: "Predictions", icon: "◈" },
    { name: "Applicants", icon: "♙" },
    { name: "Analytics", icon: "▥" },
  ];

  const filteredApplicants = applicants.filter((applicant) =>
    applicant.name.toLowerCase().includes(search.toLowerCase()) ||
    applicant.id.includes(search)
  );

  const handleFileUpload = async (event) => {
  const file = event.target.files[0];

  if (!file) return;

  event.target.value = "";

  if (!file.name.toLowerCase().endsWith(".csv")) {
    setPredictionError("Please upload a valid CSV file.");
    return;
  }

  setUploadedFile(file);
  setActivePage("Predictions");
  setPredictionError("");
  setPredictionResults([]);
  setPredictionSummary(null);
  setIsPredicting(true);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(
      "https://credit-risk-intelligence-q3se.onrender.com/predict-csv",
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
      );
    }

    setPredictionResults(data.predictions);
    setPredictionSummary(data);
    setActivePage("Predictions");
  } catch (error) {
    setPredictionError(error.message);
  } finally {
    setIsPredicting(false);
  }
};
const downloadPredictions = () => {
  if (!predictionResults.length) return;

  const headers = [
    "Applicant_ID",
    "Default_Probability",
    "Prediction",
    "Risk_Category",
  ];

  const rows = predictionResults.map((row) =>
    headers.map((header) =>
      `"${String(row[header] ?? "").replace(/"/g, '""')}"`
    ).join(",")
  );

  const csvContent = [
    headers.join(","),
    ...rows,
  ].join("\n");

  const blob = new Blob([csvContent], {
    type: "text/csv;charset=utf-8;",
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = "credit_risk_predictions.csv";
  link.click();

  URL.revokeObjectURL(url);
};

  return (
    <div className="app-container">

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">CR</div>
          <div>
            <h2>CreditRisk</h2>
            <span>INTELLIGENCE</span>
          </div>
        </div>

        <div className="sidebar-label">MENU</div>

        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${
                activePage === item.name ? "active" : ""
              }`}
              onClick={() => setActivePage(item.name)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-label">PROJECT</div>
          <div className="project-card">
            <div className="project-status"></div>
            <div>
              <h4>Credit Risk Model</h4>
              <p>XGBoost Algorithm</p>
            </div>
          </div>

          <div className="profile">
            <div className="profile-avatar">DS</div>
            <div>
              <h4>Data Scientist</h4>
              <p>Administrator</p>
            </div>
            <span className="profile-menu">•••</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">

        {/* Header */}
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / {activePage}</div>
            <h1>{activePage}</h1>
          </div>

          <div className="topbar-right">
            <div className="system-status">
              <span></span>
              Model Dashboard
            </div>
            <div className="notification">♧</div>
            <div className="top-avatar">DS</div>
          </div>
        </header>

        {/* Dashboard */}
        {activePage === "Dashboard" && (
          <>
            <section className="welcome-section">
              <div>
                <h2>Welcome to Credit Risk Intelligence</h2>
                <p>
                  Monitor applicant credit risk and explore model insights.
                </p>
              </div>
              <div className="date-label">MODEL: XGBOOST</div>
            </section>

            {/* Statistics */}
            <section className="stats-grid">
              <div className="stat-card">
                <div className="stat-top">
                  <span>Total Applicants</span>
                  <div className="stat-icon blue">♙</div>
                </div>
                <h2>48,744</h2>
                <p>Test dataset applicants</p>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span>Low Risk</span>
                  <div className="stat-icon green">✓</div>
                </div>
                <h2>21,296</h2>
                <p>43.69% of applicants</p>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span>Medium Risk</span>
                  <div className="stat-icon yellow">◷</div>
                </div>
                <h2>20,041</h2>
                <p>41.11% of applicants</p>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span>High Risk</span>
                  <div className="stat-icon red">!</div>
                </div>
                <h2>7,407</h2>
                <p>15.20% of applicants</p>
              </div>
            </section>

            {/* Analytics */}
            <section className="analytics-grid">

              <div className="panel risk-panel">
                <div className="panel-header">
                  <div>
                    <h3>Risk Distribution</h3>
                    <p>Applicant risk category breakdown</p>
                  </div>
                  <span className="panel-menu">•••</span>
                </div>

                <div className="risk-chart">
                  <div className="donut-chart">
                    <div className="donut-inner">
                      <h2>48.7K</h2>
                      <span>Applicants</span>
                    </div>
                  </div>

                  <div className="risk-legend">
                    <div>
                      <span className="legend-dot low"></span>
                      <span>Low Risk</span>
                      <strong>43.69%</strong>
                    </div>
                    <div>
                      <span className="legend-dot medium"></span>
                      <span>Medium Risk</span>
                      <strong>41.11%</strong>
                    </div>
                    <div>
                      <span className="legend-dot high"></span>
                      <span>High Risk</span>
                      <strong>15.20%</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div className="panel model-panel">
                <div className="panel-header">
                  <div>
                    <h3>Model Performance</h3>
                    <p>Validation dataset metrics</p>
                  </div>
                  <span className="model-badge">XGBoost</span>
                </div>

                <div className="performance-list">
                  <div className="performance-item">
                    <div>
                      <span>ROC-AUC</span>
                      <strong>76.95%</strong>
                    </div>
                    <div className="progress-bar">
                      <div style={{ width: "76.95%" }}></div>
                    </div>
                  </div>

                  <div className="performance-item">
                    <div>
                      <span>Precision</span>
                      <strong>24.85%</strong>
                    </div>
                    <div className="progress-bar">
                      <div style={{ width: "24.85%" }}></div>
                    </div>
                  </div>

                  <div className="performance-item">
                    <div>
                      <span>Recall</span>
                      <strong>44.49%</strong>
                    </div>
                    <div className="progress-bar">
                      <div style={{ width: "44.49%" }}></div>
                    </div>
                  </div>

                  <div className="performance-item">
                    <div>
                      <span>F1 Score</span>
                      <strong>31.89%</strong>
                    </div>
                    <div className="progress-bar">
                      <div style={{ width: "31.89%" }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Applicants Table */}
            <section className="panel applicants-panel">
              <div className="panel-header">
                <div>
                  <h3>Recent Applicants</h3>
                  <p>Latest applicant records</p>
                </div>
                <button
                  className="outline-button"
                  onClick={() => setActivePage("Applicants")}
                >
                  View All →
                </button>
              </div>

              <ApplicantTable
                applicants={applicants.slice(0, 4)}
              />
            </section>

            {/* Upload Section */}
            <section className="panel upload-panel">
              <div className="upload-content">
                <div className="upload-icon">⇧</div>
                <div>
                  <h3>Upload Applicant Dataset</h3>
                  <p>
                    Select a CSV dataset to prepare for credit risk prediction.
                  </p>
                  {uploadedFile && (
                    <p className="uploaded-name">
                      Selected: {uploadedFile.name}
                    </p>
                  )}
                </div>
              </div>

              <label className="primary-button upload-button">
                Choose CSV File
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  hidden
                />
              </label>
            </section>
          </>
        )}

        {/* Predictions Page */}
{activePage === "Predictions" && (
  <>
    <section className="panel page-panel">
      <div className="panel-header">
        <div>
          <h3>Credit Risk Predictions</h3>
          <p>
            Upload applicant data for XGBoost risk prediction.
          </p>
        </div>
      </div>

      <div className="prediction-placeholder">
        <div className="large-upload-icon">⇧</div>
        <h2>Applicant Prediction</h2>

        <p>
          Upload a compatible applicant CSV file to generate
          real-time predictions.
        </p>

        <label className="primary-button">
          {isPredicting ? "Processing..." : "Upload CSV Dataset"}

          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            disabled={isPredicting}
            hidden
          />
        </label>

        {uploadedFile && (
          <p className="uploaded-name">
            Selected file: {uploadedFile.name}
          </p>
        )}

        {isPredicting && (
          <p>Model is processing applicant records. Please wait...</p>
        )}

        {predictionError && (
          <p style={{ color: "red" }}>
            Error: {predictionError}
          </p>
        )}
      </div>
    </section>

    {predictionSummary && (
      <>
        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-top">
              <span>Total Applicants</span>
            </div>
            <h2>
              {predictionSummary.total_applicants.toLocaleString("en-IN")}
            </h2>
            <p>Processed records</p>
          </div>

          <div className="stat-card">
            <div className="stat-top">
              <span>Low Risk</span>
            </div>
            <h2>
              {predictionSummary.risk_distribution["Low Risk"].toLocaleString("en-IN")}
            </h2>
            <p>Low-risk applicants</p>
          </div>

          <div className="stat-card">
            <div className="stat-top">
              <span>Medium Risk</span>
            </div>
            <h2>
              {predictionSummary.risk_distribution["Medium Risk"].toLocaleString("en-IN")}
            </h2>
            <p>Medium-risk applicants</p>
          </div>

          <div className="stat-card">
            <div className="stat-top">
              <span>High Risk</span>
            </div>
            <h2>
              {predictionSummary.risk_distribution["High Risk"].toLocaleString("en-IN")}
            </h2>
            <p>High-risk applicants</p>
          </div>
        </section>

        <section className="panel page-panel">
          <div className="panel-header">
            <div>
              <h3>Prediction Summary</h3>
              <p>
                Average default probability:{" "}
                {(predictionSummary.average_default_probability * 100).toFixed(2)}%
              </p>
            </div>

            <button
              className="primary-button"
              onClick={downloadPredictions}
            >
              Download Results
            </button>
          </div>

          <div className="table-container">
            <table className="applicant-table">
              <thead>
                <tr>
                  <th>Applicant ID</th>
                  <th>Default Probability</th>
                  <th>Prediction</th>
                  <th>Risk Category</th>
                </tr>
              </thead>

              <tbody>
                {predictionResults.slice(0, 100).map((row, index) => (
                  <tr key={`${row.Applicant_ID}-${index}`}>
                    <td>#{row.Applicant_ID}</td>

                    <td>
                      {(row.Default_Probability * 100).toFixed(2)}%
                    </td>

                    <td>
                      {row.Prediction === 1 ? "Default" : "Non-Default"}
                    </td>

                    <td>
                      <span
                        className={`risk-badge ${
                          row.Risk_Category === "Low Risk"
                            ? "low-badge"
                            : row.Risk_Category === "Medium Risk"
                            ? "medium-badge"
                            : "high-badge"
                        }`}
                      >
                        {row.Risk_Category}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {predictionResults.length > 100 && (
            <p>
              Showing first 100 of{" "}
              {predictionResults.length.toLocaleString("en-IN")} applicants.
              Download the CSV to view all results.
            </p>
          )}
        </section>
      </>
    )}
  </>
)}


        {/* Applicants Page */}
        {activePage === "Applicants" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <h3>Applicant Records</h3>
                <p>Explore applicant risk categories.</p>
              </div>

              <input
                className="search-input"
                type="text"
                placeholder="Search applicant..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>

            <ApplicantTable applicants={filteredApplicants} />
          </section>
        )}

        {/* Analytics Page */}
        {activePage === "Analytics" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <h3>Model Analytics</h3>
                <p>Performance summary of the trained model.</p>
              </div>
            </div>

            <div className="analytics-summary">
              <div>
                <span>ROC-AUC</span>
                <h2>76.95%</h2>
              </div>
              <div>
                <span>PR-AUC</span>
                <h2>25.96%</h2>
              </div>
              <div>
                <span>F1 Score</span>
                <h2>31.89%</h2>
              </div>
              <div>
                <span>Accuracy</span>
                <h2>84.66%</h2>
              </div>
            </div>

            <p className="analytics-note">
              These metrics are from the validation dataset, not the
              unlabeled test dataset.
            </p>
          </section>
        )}

        <footer className="footer">
          <span>Credit Risk Intelligence</span>
          <span>Powered by Machine Learning & XGBoost</span>
        </footer>
      </main>
    </div>
  );
}

function ApplicantTable({ applicants }) {
  return (
    <div className="table-container">
      <table className="applicant-table">
        <thead>
          <tr>
            <th>Applicant ID</th>
            <th>Applicant</th>
            <th>Annual Income</th>
            <th>Credit Amount</th>
            <th>Risk Category</th>
          </tr>
        </thead>

        <tbody>
          {applicants.length > 0 ? (
            applicants.map((applicant) => (
              <tr key={applicant.id}>
                <td>#{applicant.id}</td>
                <td>{applicant.name}</td>
                <td>{applicant.income}</td>
                <td>{applicant.credit}</td>
                <td>
                  <span
                    className={`risk-badge ${
                      applicant.risk === "Low Risk"
                        ? "low-badge"
                        : applicant.risk === "Medium Risk"
                        ? "medium-badge"
                        : "high-badge"
                    }`}
                  >
                    {applicant.risk}
                  </span>
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="5" className="empty-table">
                No applicants found.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default App;
