// =============================================================================
// What this file does:
// PDF upload component with drag-and-drop support.
// Shows upload progress, ingestion stats after completion,
// and triggers a refresh of the DocumentList when done.
// =============================================================================

import { useState, useRef } from "react";
import { Upload, CheckCircle, AlertCircle, FileText } from "lucide-react";
import axios from "axios";

export default function UploadPanel({ onUploadComplete }) {
  const [dragging, setDragging]   = useState(false);  // drag-over state
  const [uploading, setUploading] = useState(false);  // upload in progress
  const [result, setResult]       = useState(null);   // ingestion result
  const [error, setError]         = useState(null);   // error message
  const [progress, setProgress]   = useState("");     // progress text
  const fileInputRef              = useRef(null);     // hidden file input ref

  const handleFile = async (file) => {
    if (!file) return;

    // Validate PDF
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setUploading(true);       // show uploading state
    setError(null);           // clear previous error
    setResult(null);          // clear previous result
    setProgress("Uploading PDF...");

    try {
      const formData = new FormData();      // create multipart form
      formData.append("file", file);        // attach the PDF file

      setProgress("Extracting text and creating chunks...");

      const res = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },  // required for file upload
        timeout: 300000,      // 5 minute timeout — large PDFs take time to embed
      });

      setResult(res.data);    // store ingestion result
      setProgress("");
      onUploadComplete();     // tell parent to refresh DocumentList

    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Upload failed. Make sure the FastAPI server is running."
      );
      setProgress("");
    } finally {
      setUploading(false);    // stop uploading state
    }
  };

  // Handle file selected via file picker
  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>

      {/* Drop zone */}
      <div
        onClick={() => !uploading && fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `1px dashed ${dragging ? "#1a1a1a" : "#DDD8CC"}`,
          borderRadius: "4px",
          padding: "2rem 1rem",
          textAlign: "center",
          cursor: uploading ? "not-allowed" : "pointer",
          backgroundColor: dragging ? "#EDE8DC" : "#FAFAF7",
          transition: "all 0.2s ease",
        }}
      >
        {uploading ? (
          // Uploading state
          <div>
            <div style={{
              width: "2rem", height: "2rem",
              border: "2px solid #DDD8CC",
              borderTopColor: "#1a1a1a",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              margin: "0 auto 0.75rem",
            }} />
            <p style={{ fontSize: "0.75rem", color: "#1a1a1a", fontWeight: 500, margin: 0 }}>
              {progress}
            </p>
            <p style={{ fontSize: "0.65rem", color: "#888888", marginTop: "0.25rem" }}>
              This may take 1–3 minutes for large documents
            </p>
          </div>
        ) : (
          // Default state
          <div>
            <Upload size={20} style={{ color: "#C8C2B4", margin: "0 auto 0.75rem" }} />
            <p style={{ fontSize: "0.75rem", color: "#1a1a1a", fontWeight: 500, margin: 0 }}>
              Drop a PDF here or click to browse
            </p>
            <p style={{ fontSize: "0.65rem", color: "#888888", marginTop: "0.25rem" }}>
              Basel III, MiFID II, IFRS 9, FCA Handbook, SEC filings...
            </p>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileInput}
        style={{ display: "none" }}
      />

      {/* Success result */}
      {result && (
        <div style={{
          padding: "0.75rem 1rem",
          backgroundColor: "#F0FAF4",
          border: "1px solid #C8E6D4",
          borderRadius: "4px",
          display: "flex", gap: "0.75rem", alignItems: "flex-start",
        }}>
          <CheckCircle size={15} style={{ color: "#2D7A4F", flexShrink: 0, marginTop: "1px" }} />
          <div>
            <p style={{ fontSize: "0.75rem", color: "#1a1a1a", fontWeight: 500, margin: 0 }}>
              {result.filename} ingested successfully
            </p>
            <p style={{ fontSize: "0.65rem", color: "#555555", marginTop: "2px" }}>
              {result.pages} pages · {result.chunks} chunks stored in ChromaDB
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: "0.75rem 1rem",
          backgroundColor: "#FDF0F0",
          border: "1px solid #E8C8C8",
          borderRadius: "4px",
          display: "flex", gap: "0.75rem", alignItems: "flex-start",
        }}>
          <AlertCircle size={15} style={{ color: "#8B2020", flexShrink: 0, marginTop: "1px" }} />
          <p style={{ fontSize: "0.75rem", color: "#8B2020", margin: 0 }}>{error}</p>
        </div>
      )}

      {/* Spinning keyframe — injected inline */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}