# Security Policy & Assessment

## Security Posture

The **House Price Prediction System** prioritizes input validation, safe model serialization, and secure deployment practices.

### 1. Model Deserialization Security
- Standard Python `pickle`/`joblib` deserialization carries inherent code execution risks if an untrusted binary is loaded.
- In this repository:
  - Models are serialized to a restricted local `models/` directory.
  - Every model artifact is accompanied by a cryptographic SHA-256 checksum in `models/model_metadata.json`.
  - When loading models, checksums are available to verify file integrity.

### 2. Input Sanitation & Boundary Validation
- All prediction requests (single-property or batch CSV) pass through `HousePricePredictor.validate_and_normalize_input()`.
- Numerical bounds (`area`, `bedrooms`, `bathrooms`, `age`, `parking`) are enforced against strict domain rules defined in `src/config.py`.
- Categorical options (`locality`, `furnished_status`) are whitelisted; unrecognized categories raise immediate validation errors.
- **Physical Floor Clamping**: Negative price valuations are blocked at the domain layer, clamping outputs to a minimum replacement floor ($₹5.0\text{ Lakhs}$).

### 3. Secret Management & Version Control Hygiene
- No credentials, tokens, or private API keys are stored in the repository.
- A standardized `.gitignore` is enforced to prevent accidental commits of `.env` files, virtual environments (`.venv/`), and cache files.
- An environment template `.env.example` is provided for local deployment configuration.

### 4. Reporting Security Issues
If you identify a security vulnerability or insecure dependency, please report it responsibly by contacting the maintainer or opening a private security advisory on GitHub.
