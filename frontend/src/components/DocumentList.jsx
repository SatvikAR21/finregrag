// =============================================================================
// What this file does:
// Fetches and displays all documents currently stored in ChromaDB.
// Shows filename, chunk count, and status badge for each document.
// Refreshes automatically after a new upload completes.
// =============================================================================

import { useState, useEffect } from "react";
import { FileText, RefreshCw } from "lucide-react";
import axios from "axios";

export default function DocumentList({ refreshTrigger }) {
  const [docs, setDocs]       = useState([]);       // list of ingested documents
  const [loading, setLoading] = useState(true);     // loading state

  const fetchDocs = async () => {
    setLoading(true);
    try {
      const res = await axios.get("/api/documents"); // GET /documents
      setDocs(res.data.documents || []);             // store document list
    } catch {
      setDocs([]);                                   // on error show empty
    } finally {
      setLoading(false);
    }
  };

  // Fetch on mount and whenever a new upload completes
  useEffect(() => { fetchDocs(); }, [refreshTrigger]);

  return (
    <div style={{
      backgroundColor: "#FAFAF7",
      border: "1px solid #DDD8CC",
      borderRadius: "4px",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "0.75rem 1rem",
        borderBottom: "1px solid #EDE8DC",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{
          fontSize: "0.6rem", letterSpacing: "0.2em",
          fontWeight: 600, color: "#888888",
        }}>
          INGESTED DOCUMENTS
        </span>
        <button onClick={fetchDocs} style={{
          background: "none", border: "none",
          cursor: "pointer", color: "#888888",
          display: "flex", alignItems: "center",
        }}>
          <RefreshCw size={11} />
        </button>
      </div>

      {/* Document list */}
      {loading ? (
        <div style={{ padding: "1rem", textAlign: "center" }}>
          <p style={{ fontSize: "0.7rem", color: "#C8C2B4" }}>Loading...</p>
        </div>
      ) : docs.length === 0 ? (
        <div style={{ padding: "1rem", textAlign: "center" }}>
          <p style={{ fontSize: "0.7rem", color: "#C8C2B4" }}>No documents ingested yet.</p>
        </div>
      ) : (
        <div>
          {docs.map((doc, i) => (
            <div key={doc.filename} style={{
              padding: "0.75rem 1rem",
              borderBottom: i < docs.length - 1 ? "1px solid #EDE8DC" : "none",
              display: "flex", alignItems: "center", gap: "0.75rem",
            }}>
              {/* Icon */}
              <FileText size={13} style={{ color: "#C8C2B4", flexShrink: 0 }} />

              {/* Doc info */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{
                  fontSize: "0.75rem", color: "#1a1a1a",
                  fontWeight: 500, margin: 0,
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {doc.filename}
                </p>
                <p style={{ fontSize: "0.65rem", color: "#888888", margin: "2px 0 0" }}>
                  {doc.chunks} chunks
                </p>
              </div>

              {/* Status badge */}
              <span style={{
                fontSize: "0.55rem", letterSpacing: "0.1em",
                padding: "2px 6px", borderRadius: "2px",
                backgroundColor: "#1a1a1a", color: "#F5F0E8",
                fontWeight: 600, flexShrink: 0,
              }}>
                READY
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}