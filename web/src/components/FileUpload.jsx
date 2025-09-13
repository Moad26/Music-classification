import { useCallback, useState } from "react";
import "./FileUpload.css";
import { Music } from "lucide-react";

const FileUpload = ({ onFileSelect }) => {
  const [dragActive, setDragActive] = useState(false);
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);
  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFileSelect(e.dataTransfer.files[0]);
      }
    },
    [onFileSelect],
  );
  const handleChange = useCallback(
    (e) => {
      e.preventDefault();

      if (e.target.files && e.target.files[0]) {
        onFileSelect(e.target.files[0]);
      }
    },
    [onFileSelect],
  );
  return (
    <div
      className={`file-upload ${dragActive ? "drag-active" : ""}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        id="file-input"
        className="file-input"
        accept="audio/*"
        onChange={handleChange}
      />
      <lable htmlFor="file-input" className="file-lable">
        <div className="upload-icon">
          <Music />
        </div>
        <p>Drag & drop your audio file here</p>
        <p className="subtext">or click to browse</p>
      </lable>
    </div>
  );
};

export default FileUpload;
