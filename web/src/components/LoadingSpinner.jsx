import "./LoadingSpinner.css";
const LoadingSpinner = () => {
  return (
    <div className="loading-spinner">
      <div className="spinner"></div>
      <p>Analyzing audio...</p>
    </div>
  );
};
export default LoadingSpinner;
