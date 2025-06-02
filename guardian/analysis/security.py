import subprocess
import json
import sys
import os
import logging
from typing import Dict, Any, List, Generator, Tuple

logger = logging.getLogger(__name__)

def check_dependencies_vulnerabilities() -> Dict[str, Any]:
    """
    Checks for vulnerabilities in the current Python environment's installed packages using 'safety'.
    Returns:
        list: A list of vulnerability dictionaries found by safety, or an empty list if none are found or an error occurs.
              Each dictionary typically contains details like:
              package_name, affected_versions, vulnerability_id, description, more_info_url, etc.
    """
    vulnerabilities_found = []
    try:
        # Construct the command to run safety and get JSON output
        # We need to use the pip from the current virtual environment
        # sys.executable gives the path to the current Python interpreter
        # safety check --json --output stdout
        # For simplicity, we'll assume 'safety' is in the PATH of the .venv
        # If guardian is run via .venv/bin/guardian, then .venv/bin should be in PATH for subprocesses.
        
        # A more robust way to call safety from the same venv:
        # Find safety executable relative to sys.executable
        import os
        venv_bin_path = os.path.dirname(sys.executable)
        safety_executable = os.path.join(venv_bin_path, "safety")

        if not os.path.exists(safety_executable):
            # Fallback if not found directly, try just 'safety' hoping it's in PATH
            safety_executable = "safety" 
            print(f"Warning: Could not find {safety_executable} directly in venv bin. Trying 'safety' from PATH.")


        # Using --stdin to pass installed packages, as 'safety check' without it might scan the cwd.
        # First, get installed packages using pip freeze.
        pip_freeze_cmd = [os.path.join(venv_bin_path, "pip"), "freeze"]
        pip_process = subprocess.run(pip_freeze_cmd, capture_output=True, text=True, check=False)

        if pip_process.returncode != 0:
            print(f"Error running 'pip freeze': {pip_process.stderr}")
            return {"error": "Failed to list installed packages.", "details": []}

        installed_packages = pip_process.stdout
        
        # Now run safety check with the list of packages via stdin
        # Removed "--output", "stdout" as --json implies output to stdout if no file is specified.
        cmd = [safety_executable, "check", "--stdin", "--json"]
        process = subprocess.run(cmd, input=installed_packages, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            # Safety exits with 0 if no vulnerabilities are found, or if outputting JSON.
            # If vulnerabilities are found and output is not JSON, it exits with a non-zero code.
            # With --json, it should always exit 0 unless there's a major error in safety itself.
            try:
                # The output is a stream of JSON objects, one per line, if multiple vulnerabilities.
                # However, the typical JSON output for 'safety check --json' is a single JSON object (a list of vulns).
                # Let's try to parse it as a single JSON object first.
                # The output from safety with --json is a list of vulnerabilities.
                # Example: [{"package": "requests", ...}, {"package": "django", ...}]
                # If no vulns, it's an empty list "[]".
                # If there's an error in safety itself, it might print non-JSON.
                
                # Safety's JSON output is a list of vulnerabilities.
                # Each item in the list is a dictionary.
                # [
                #   ["requests", "<2.25.0", "43591", "CVE-...", "...", "...", "..."],
                #   ["django", "<...", "...", ... ]
                # ]
                # The actual structure is more like:
                # [
                #   {
                #     "package_name": "requests",
                #     "affected_versions": "<2.25.0",
                #     "vulnerability_id": "43591",
                #     "description": "...",
                #     ...
                #   }
                # ]
                # Let's adjust based on typical safety JSON output.
                # The output is actually a list of lists, not list of dicts directly from stdout.
                # [["package_name", "spec", "version", "vuln_id", "advisory", "cvss_v2", "cvss_v3", "severity", "more_info_url"]]
                # This is the header. Then data rows.
                # This is for --output bare. For --output json, it's different.
                
                # For `safety check --json --output stdout`:
                # It outputs a JSON array of vulnerability objects.
                # Example:
                # [
                #   {
                #     "name": "requests",
                #     "spec": "<2.20.0",
                #     "version": "2.19.1",
                #     "advisory": "Requests...",
                #     "id": "37698",
                #     "cvssv2": null,
                #     "cvssv3": null
                #   }
                # ]
                # If no vulnerabilities, it's an empty list `[]`.
                
                # Extract the JSON part of the output, ignoring potential leading text (like deprecation warnings)
                json_output_str = process.stdout
                first_brace = json_output_str.find('{')
                first_bracket = json_output_str.find('[')

                start_index = -1

                if first_brace != -1 and first_bracket != -1:
                    start_index = min(first_brace, first_bracket)
                elif first_brace != -1:
                    start_index = first_brace
                elif first_bracket != -1:
                    start_index = first_bracket
                
                if start_index != -1:
                    # Find the matching end for the JSON structure
                    # Assuming the main JSON output is an object, so it ends with '}'
                    # If it could be an array, we'd also check for ']'
                    # For safety's output, it's a JSON object.
                    last_brace = json_output_str.rfind('}')
                    if last_brace > start_index : # Ensure '}' is after '{'
                        json_data_str = json_output_str[start_index : last_brace + 1]
                        try:
                            parsed_output = json.loads(json_data_str)
                            
                            # Safety 3.x with --json outputs a single JSON object containing a "vulnerabilities" list
                            if isinstance(parsed_output, dict) and "vulnerabilities" in parsed_output:
                                vulnerabilities_found = parsed_output["vulnerabilities"]
                            # The output might also be a direct list of vulnerabilities in some contexts/versions
                            elif isinstance(parsed_output, list):
                                vulnerabilities_found = parsed_output
                            else:
                                print(f"Warning: 'safety check --json' output was not in the expected dict or list format: {type(parsed_output)}")
                        except json.JSONDecodeError as e_inner:
                            print(f"Error decoding extracted JSON from safety: {e_inner}\nExtracted data was:\n{json_data_str[:500]}...") # Print first 500 chars of attempted parse
                            return {"error": "Failed to decode extracted JSON from safety.", "details": []}
                    else:
                        print(f"Error: Could not find end of JSON object in safety output: {process.stdout}")
                        return {"error": "Could not find end of JSON object in safety output.", "details": []}
                else:
                    print(f"Error: Could not find start of JSON in safety output: {process.stdout}")
                    return {"error": "Could not find JSON in safety output.", "details": []}

            except json.JSONDecodeError as e: # This outer except might not be reached if inner one catches
                print(f"Error decoding JSON from safety: {e}\nOutput was:\n{process.stdout}")
                return {"error": "Failed to decode JSON from safety.", "details": []}
        else:
            # Safety might exit non-zero if vulnerabilities are found AND not using --json,
            # or if there's an operational error.
            # Since we use --json, a non-zero exit here likely means an error with safety itself.
            print(f"Error running safety: {process.stderr}")
            return {"error": f"Safety tool error (exit code {process.returncode})", "details": [], "stderr": process.stderr}

    except FileNotFoundError:
        print(f"Error: 'safety' command not found. Make sure it's installed and in your PATH.")
        return {"error": "'safety' command not found.", "details": []}
    except Exception as e:
        print(f"An unexpected error occurred while checking for vulnerabilities: {e}")
        return {"error": f"Unexpected error: {str(e)}", "details": []}
    
    return {"error": None, "details": vulnerabilities_found}

def check_for_eval_usage(code_content):
    """
    Checks for the use of 'eval(' in the given code content.
    Args:
        code_content (str): The source code as a string.
    Returns:
        list: A list of dictionaries, each containing 'line_number' and 'line_content'
              where 'eval(' was found.
    """
    findings = []
    import re
    for i, line in enumerate(code_content.splitlines()):
        if re.search(r'\beval\s*\(', line):
            if not line.lstrip().startswith('#'):
                findings.append({"line_number": i + 1, "line_content": line.strip()})
    return findings

def check_for_hardcoded_secrets(code_content):
    """
    Checks for potential hardcoded secrets in the given code content.
    Args:
        code_content (str): The source code as a string.
    Returns:
        list: A list of dictionaries, each containing 'line_number', 'line_content',
              and 'pattern_name' for each potential secret found.
    """
    findings = []
    import re

    # Common keywords often associated with secrets.
    # This list can be expanded. Case-insensitive matching for keywords.
    # Regex looks for: keyword, optional _KEY or _TOKEN or _SECRET, then assignment to a string.
    secret_patterns = {
        "API_KEY": r"""(['"]\s*API_KEY\s*['"]\s*[:=]|API_KEY\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "SECRET_KEY": r"""(['"]\s*SECRET_KEY\s*['"]\s*[:=]|SECRET_KEY\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "ACCESS_KEY": r"""(['"]\s*ACCESS_KEY\s*['"]\s*[:=]|ACCESS_KEY\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "PASSWORD": r"""(['"]\s*PASSWORD\s*['"]\s*[:=]|PASSWORD\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "TOKEN": r"""(['"]\s*TOKEN\s*['"]\s*[:=]|TOKEN\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "AUTH_TOKEN": r"""(['"]\s*AUTH_TOKEN\s*['"]\s*[:=]|AUTH_TOKEN\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        "SECRET": r"""(['"]\s*SECRET\s*['"]\s*[:=]|SECRET\s*[:=])\s*['"](?P<secret_value>[^'"]+)['"]""",
        # Generic pattern for variable names ending with _KEY, _TOKEN, _SECRET
        "GENERIC_KEY": r"""\b[A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWD|PASSWORD)\b\s*[:=]\s*['"](?P<secret_value>[^'"]{4,})['"]""" # at least 4 chars
    }
    
    # More specific patterns to reduce false positives (e.g. for common short strings)
    # Example: AWS Access Key ID (20 char, uppercase alphanumeric)
    # aws_access_key_pattern = r"""(['"]\s*[Aa][Ww][Ss]_[Aa][Cc][Cc][Ee][Ss][Ss]_[Kk][Ee][Yy]_[Ii][Dd]\s*['"]\s*[:=]|AWSAccessKeyId\s*[:=])\s*['"](?P<secret_value>[A-Z0-9]{20})['"]"""
    # Example: AWS Secret Access Key (40 char, mixed case alphanumeric + /+=)
    # aws_secret_key_pattern = r"""(['"]\s*[Aa][Ww][Ss]_[Ss][Ee][Cc][Rr][Ee][Tt]_[Aa][Cc][Cc][Ee][Ss][Ss]_[Kk][Ee][Yy]\s*['"]\s*[:=]|AWSSecretKey\s*[:=])\s*['"](?P<secret_value>[A-Za-z0-9/+=]{40})['"]"""

    # For v1, let's use the broader patterns first.
    # We should be careful about matching too broadly (e.g. "key = 'value'" could be anything)

    for i, line in enumerate(code_content.splitlines()):
        if line.lstrip().startswith('#'): # Skip full-line comments
            continue
        for pattern_name, pattern_regex in secret_patterns.items():
            match = re.search(pattern_regex, line, re.IGNORECASE) # Ignore case for keywords
            if match:
                # Basic check to avoid matching empty strings or very short, common strings if not specific enough
                secret_val_match = match.groupdict().get('secret_value')
                if secret_val_match and len(secret_val_match) < 4 and pattern_name == "GENERIC_KEY": # Avoid short generic matches
                    continue
                if secret_val_match and (secret_val_match.lower() in ["none", "null", "true", "false", "''", '""']): # Avoid common non-secrets
                    continue

                findings.append({
                    "line_number": i + 1,
                    "line_content": line.strip(),
                    "pattern_name": pattern_name,
                    "matched_value_preview": secret_val_match[:10] + "..." if secret_val_match and len(secret_val_match) > 10 else secret_val_match
                })
                break # Found a pattern on this line, move to next line
    return findings


# Helper to walk project files
def _walk_project_files(project_path: str) -> Generator[Tuple[str, str], None, None]:
    for root, _, files in os.walk(project_path):
        # Common directories to skip
        if any(skip_dir in root for skip_dir in ['.venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'build', 'dist', 'target', 'out']):
            continue
        for file_name in files:
            if file_name.endswith(".py"): # For now, only Python files for eval/secrets
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    yield file_path, content
                except Exception as e:
                    logger.warning(f"Could not read file {file_path} for security scan: {e}")

DEFAULT_SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.2,
    "UNKNOWN": 0.3, # Default if severity is not parseable or not present
}
EVAL_USAGE_WEIGHT = 1.0
HARDCODED_SECRET_WEIGHT = 0.7
# If total weighted score reaches this, density is considered 1.0
# This value can be made configurable if needed.
VULNERABILITY_NORMALIZATION_DENOMINATOR = 10.0

def get_raw_weighted_vulnerability_density(project_path: str, config: Dict[str, Any] = None) -> float:
    """
    Calculates a raw weighted vulnerability density score (0-1) for the project.
    A score of 0 means low/no vulnerability impact, 1 means high impact/density.

    Args:
        project_path: The root path of the project to analyze.
        config: Sensor-specific configuration (currently unused, for future expansion).

    Returns:
        A float between 0.0 and 1.0 representing the weighted vulnerability density.
    """
    if config is None:
        config = {}

    total_weighted_score = 0.0
    findings_summary = {
        "dependencies": 0,
        "eval_usage": 0,
        "hardcoded_secrets": 0
    }

    # 1. Dependency Vulnerabilities
    logger.info("Checking for dependency vulnerabilities...")
    dep_vuln_result = check_dependencies_vulnerabilities()
    if dep_vuln_result.get("error"):
        logger.warning(f"Could not check dependency vulnerabilities: {dep_vuln_result['error']}")
        # Optional: Add a penalty if checks fail. For now, we just log.
        # total_weighted_score += config.get("dependency_check_failure_penalty", 0.5)
    elif dep_vuln_result.get("details"):
        findings_summary["dependencies"] = len(dep_vuln_result["details"])
        for vuln in dep_vuln_result["details"]:
            severity_str = "UNKNOWN" # Default
            
            # Try to get severity from CVSSv3 (preferred), then CVSSv2
            cvss_data = vuln.get("cvssv3") # safety output often has this as a dict
            if isinstance(cvss_data, dict) and cvss_data.get("base_severity"):
                severity_str = cvss_data.get("base_severity", "UNKNOWN").upper()
            else: # Fallback or other formats safety might use
                cvss_data_v2 = vuln.get("cvssv2")
                if isinstance(cvss_data_v2, dict) and cvss_data_v2.get("base_severity"):
                     severity_str = cvss_data_v2.get("base_severity", "UNKNOWN").upper()
                elif 'severity' in vuln and isinstance(vuln['severity'], str): # Direct severity field
                    severity_str = vuln['severity'].upper()
                # Some safety versions might put severity directly in a field like 'severity_rating'
                elif vuln.get('severity_rating') and isinstance(vuln['severity_rating'], str):
                    severity_str = vuln['severity_rating'].upper()


            weight = DEFAULT_SEVERITY_WEIGHTS.get(severity_str, DEFAULT_SEVERITY_WEIGHTS["UNKNOWN"])
            total_weighted_score += weight
            logger.debug(f"Dependency vuln: {vuln.get('package_name', vuln.get('name', 'N/A'))}, Severity: {severity_str}, Weight: {weight:.2f}")

    # 2. Eval Usage and Hardcoded Secrets in project files
    logger.info(f"Scanning project files in '{project_path}' for eval usage and hardcoded secrets...")
    files_scanned_count = 0
    if os.path.isdir(project_path):
        for file_path, content in _walk_project_files(project_path):
            files_scanned_count += 1
            
            eval_findings = check_for_eval_usage(content)
            for _ in eval_findings:
                total_weighted_score += EVAL_USAGE_WEIGHT
                findings_summary["eval_usage"] += 1
                logger.debug(f"Eval usage found in {file_path}")

            secret_findings = check_for_hardcoded_secrets(content)
            for finding in secret_findings:
                total_weighted_score += HARDCODED_SECRET_WEIGHT
                findings_summary["hardcoded_secrets"] += 1
                logger.debug(f"Hardcoded secret '{finding.get('pattern_name')}' found in {file_path} (Line: {finding.get('line_number')})")
    else:
        logger.warning(f"Project path '{project_path}' is not a directory. Skipping file scan for eval/secrets.")

    if files_scanned_count == 0 and os.path.isdir(project_path):
        logger.warning(f"No Python files found to scan for eval/secrets in {project_path}")

    # Normalize the total_weighted_score to 0-1 range
    normalization_denominator = config.get("vulnerability_normalization_denominator", VULNERABILITY_NORMALIZATION_DENOMINATOR)
    if normalization_denominator <= 0: # Prevent division by zero or negative
        normalization_denominator = VULNERABILITY_NORMALIZATION_DENOMINATOR
        logger.warning(f"Invalid 'vulnerability_normalization_denominator' in config. Using default: {VULNERABILITY_NORMALIZATION_DENOMINATOR}")

    normalized_density = min(1.0, total_weighted_score / normalization_denominator)
    
    logger.info(
        f"Vulnerability Density Calculation: "
        f"Total Weighted Score = {total_weighted_score:.2f} (Normalized against {normalization_denominator}), "
        f"Normalized Density = {normalized_density:.3f}. "
        f"Findings: Dependencies={findings_summary['dependencies']}, Eval Usage={findings_summary['eval_usage']}, Hardcoded Secrets={findings_summary['hardcoded_secrets']}"
    )
    return normalized_density


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger.info("--- Dependency Vulnerability Check ---")
    # Ensure safety is installed in the environment this script is run with.
    # For testing, you might need a project with known vulnerable dependencies.
    dep_results = check_dependencies_vulnerabilities()
    if dep_results.get("error"):
        print(f"Error: {dep_results['error']}")
        if "stderr" in dep_results:
            print(f"Stderr: {dep_results['stderr']}")
    elif dep_results.get("details"):
        print(f"Found {len(dep_results['details'])} vulnerabilities:")
        for vuln in dep_results['details']:
            # Adjusting print based on common 'safety' JSON output structure
            pkg_name = vuln.get('package_name', vuln.get('name', 'N/A'))
            version = vuln.get('analyzed_version', vuln.get('version', 'N/A'))
            vuln_id = vuln.get('vulnerability_id', vuln.get('id', 'N/A'))
            advisory = vuln.get('advisory', 'N/A')
            print(f"  - Package: {pkg_name} ({version}) - ID: {vuln_id} - Advisory: {advisory[:60]}...")
    else:
        logger.info("No dependency vulnerabilities found or 'details' key missing.")

    logger.info("\n--- Eval Usage Check ---")
    sample_code_with_eval = """
import os
def dangerous_function(user_input):
    # This is a comment eval("test")
    result = eval(user_input) # Potential security risk
    print(result)
    eval ("another one")
    safe_eval_like_name = "evaluation"
"""
    eval_findings = check_for_eval_usage(sample_code_with_eval)
    if eval_findings:
        logger.info(f"Found {len(eval_findings)} uses of 'eval(':")
        for finding in eval_findings:
            logger.info(f"  - Line {finding['line_number']}: {finding['line_content']}")
    else:
        logger.info("No uses of 'eval(' found in sample code.")

    sample_code_without_eval = """
def safe_function(x):
    return x * 2
"""
    eval_findings_safe = check_for_eval_usage(sample_code_without_eval)
    if eval_findings_safe:
        logger.warning(f"Found {len(eval_findings_safe)} uses of 'eval(' in safe sample (UNEXPECTED):")
        for finding in eval_findings_safe:
            logger.warning(f"  - Line {finding['line_number']}: {finding['line_content']}")
    else:
        logger.info("No uses of 'eval(' found in safe sample code (as expected).")

    logger.info("\n--- Hardcoded Secrets Check ---")
    sample_code_with_secrets = """
API_KEY = "12345abcdef12345abcdef" # a secret
MY_PASSWORD = 'supersecretpassword123!'
config = {"SECRET": "another_one_here"}
# NOT_A_SECRET_KEY = "some_public_value"
# token = os.environ.get("MY_APP_TOKEN") # Good
# api_key = settings.API_KEY # Good
# password = get_password_from_vault() # Good
# empty_secret = ""
# short_key = "key"
"""
    secret_findings = check_for_hardcoded_secrets(sample_code_with_secrets)
    if secret_findings:
        logger.info(f"Found {len(secret_findings)} potential hardcoded secrets:")
        for finding in secret_findings:
            logger.info(f"  - Line {finding['line_number']}: Pattern: {finding['pattern_name']} - Content: {finding['line_content']} (Value: {finding['matched_value_preview']})")
    else:
        logger.info("No potential hardcoded secrets found in sample code.")

    logger.info("\n--- Raw Weighted Vulnerability Density Check ---")
    # Create a dummy project structure for testing
    dummy_project_path = ".tmp_dummy_security_project"
    if not os.path.exists(dummy_project_path):
        os.makedirs(dummy_project_path)
    
    with open(os.path.join(dummy_project_path, "app.py"), "w") as f:
        f.write("API_KEY = 'secret123'\n")
        f.write("result = eval('1+1')\n")
        f.write("eval('2+2') # Another eval\n")

    with open(os.path.join(dummy_project_path, "utils.py"), "w") as f:
        f.write("PASSWORD = 'pass'\n") # Short secret, might be filtered by some rules but pattern matches

    # To test dependency scan, you'd typically run this in an environment
    # where 'safety' can find vulnerable packages.
    # For this standalone test, dependency scan might return empty or error if safety not setup.
    density = get_raw_weighted_vulnerability_density(dummy_project_path)
    logger.info(f"Calculated Raw Weighted Vulnerability Density for '{dummy_project_path}': {density:.3f}")

    # Clean up dummy project
    import shutil
    if os.path.exists(dummy_project_path):
        try:
            shutil.rmtree(dummy_project_path)
            logger.info(f"Cleaned up dummy project: {dummy_project_path}")
        except Exception as e:
            logger.error(f"Could not clean up dummy project {dummy_project_path}: {e}")