import { useState } from "react";
import QueryPanel from "./components/QueryPanel";
import AnswerDisplay from "./components/AnswerDisplay";
import SourcesPanel from "./components/SourcesPanel";
import AnalyticsDashboard from "./components/AnalyticsDashboard";
import UploadPanel from "./components/UploadPanel";
import DocumentList from "./components/DocumentList";

export default function App() {
  const [result, setResult]             = useState(null);
  const [activeTab, setActiveTab]       = useState("query");
  const [uploadRefresh, setUploadRefresh] = useState(0);  // increments to trigger refresh

  // Called after a successful upload — refreshes DocumentList
  const handleUploadComplete = () => setUploadRefresh(n => n + 1);

  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F5F0E8" }}>

      {/* ── HEADER ── */}
      <header style={{
        backgroundColor: "#F5F0E8",
        borderBottom: "1px solid #DDD8CC",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div className="max-w-6xl mx-auto px-8 py-5 flex items-center justify-between">

          {/* Wordmark */}
          <div>
            <h1 style={{
              fontFamily: "'Playfair Display', serif",
              fontSize: "1.1rem", fontWeight: 500,
              color: "#0d0d0d", letterSpacing: "0.05em", lineHeight: 1,
            }}>
              FINREG
            </h1>
            <p style={{
              fontSize: "0.6rem", letterSpacing: "0.2em",
              color: "#888888", marginTop: "3px", fontWeight: 500,
            }}>
              REGULATORY INTELLIGENCE
            </p>
          </div>

          {/* Nav tabs */}
          <nav style={{ display: "flex", gap: "0.25rem" }}>
            {[
              { key: "query",     label: "QUERY" },
              { key: "documents", label: "DOCUMENTS" },
              { key: "analytics", label: "ANALYTICS" },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  fontSize: "0.65rem", letterSpacing: "0.18em", fontWeight: 600,
                  padding: "0.5rem 1.25rem",
                  border: activeTab === tab.key ? "1px solid #1a1a1a" : "1px solid transparent",
                  backgroundColor: activeTab === tab.key ? "#1a1a1a" : "transparent",
                  color: activeTab === tab.key ? "#F5F0E8" : "#888888",
                  cursor: "pointer", transition: "all 0.2s ease", borderRadius: "2px",
                }}
                onMouseEnter={e => {
                  if (activeTab !== tab.key) {
                    e.currentTarget.style.color = "#1a1a1a";
                    e.currentTarget.style.borderColor = "#C8C2B4";
                  }
                }}
                onMouseLeave={e => {
                  if (activeTab !== tab.key) {
                    e.currentTarget.style.color = "#888888";
                    e.currentTarget.style.borderColor = "transparent";
                  }
                }}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Live indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <div className="pulse-ink" style={{
              width: "5px", height: "5px",
              borderRadius: "50%", backgroundColor: "#1a1a1a",
            }} />
            <span style={{ fontSize: "0.6rem", letterSpacing: "0.15em", color: "#888888", fontWeight: 500 }}>
              LIVE
            </span>
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT ── */}
      <main className="max-w-6xl mx-auto px-8 py-10">

        {/* ── QUERY TAB ── */}
        {activeTab === "query" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: "3rem" }}>

            {/* Left: query + answer */}
            <div>
              <div style={{ marginBottom: "2rem" }}>
                <span className="section-rule" />
                <h2 style={{
                  fontFamily: "'Playfair Display', serif",
                  fontSize: "1.5rem", fontWeight: 500,
                  color: "#0d0d0d", lineHeight: 1.3, marginBottom: "0.4rem",
                }}>
                  Regulatory Query
                </h2>
                <p style={{ fontSize: "0.75rem", color: "#888888", letterSpacing: "0.03em" }}>
                  Search across all ingested regulatory documents simultaneously.
                </p>
              </div>
              <QueryPanel onResult={setResult} />
              {result && <div style={{ marginTop: "2rem" }}><AnswerDisplay result={result} /></div>}
            </div>

            {/* Right: sources */}
            <div style={{ paddingTop: "4.5rem" }}>
              <SourcesPanel sources={result?.sources} />
            </div>
          </div>
        )}

        {/* ── DOCUMENTS TAB ── */}
        {activeTab === "documents" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3rem" }}>

            {/* Upload section */}
            <div>
              <div style={{ marginBottom: "2rem" }}>
                <span className="section-rule" />
                <h2 style={{
                  fontFamily: "'Playfair Display', serif",
                  fontSize: "1.5rem", fontWeight: 500,
                  color: "#0d0d0d", lineHeight: 1.3, marginBottom: "0.4rem",
                }}>
                  Upload Document
                </h2>
                <p style={{ fontSize: "0.75rem", color: "#888888" }}>
                  Upload any regulatory PDF to make it immediately queryable.
                </p>
              </div>
              <UploadPanel onUploadComplete={handleUploadComplete} />
            </div>

            {/* Document list section */}
            <div>
              <div style={{ marginBottom: "2rem" }}>
                <span className="section-rule" />
                <h2 style={{
                  fontFamily: "'Playfair Display', serif",
                  fontSize: "1.5rem", fontWeight: 500,
                  color: "#0d0d0d", lineHeight: 1.3, marginBottom: "0.4rem",
                }}>
                  Document Library
                </h2>
                <p style={{ fontSize: "0.75rem", color: "#888888" }}>
                  All documents currently indexed and ready to query.
                </p>
              </div>
              <DocumentList refreshTrigger={uploadRefresh} />
            </div>
          </div>
        )}

        {/* ── ANALYTICS TAB ── */}
        {activeTab === "analytics" && <AnalyticsDashboard />}
      </main>

      {/* ── FOOTER ── */}
      <footer style={{ borderTop: "1px solid #DDD8CC", padding: "2rem 0", marginTop: "4rem" }}>
        <div className="max-w-6xl mx-auto px-8 flex items-center justify-between">
          <p style={{ fontSize: "0.6rem", letterSpacing: "0.15em", color: "#C8C2B4" }}>
            © 2026 FINREG-RAG · FINANCIAL REGULATORY INTELLIGENCE
          </p>
          <p style={{ fontSize: "0.6rem", letterSpacing: "0.15em", color: "#C8C2B4" }}>
            BASEL III · HYBRID RAG · MULTI-DOCUMENT
          </p>
        </div>
      </footer>
    </div>
  );
}