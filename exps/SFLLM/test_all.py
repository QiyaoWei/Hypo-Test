#!/usr/bin/env python3
"""
Test script to check if all Python files in SFLLM directory run without import/syntax errors
"""
import sys
import ast
import importlib.util
import os
from pathlib import Path

def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def check_imports(filepath):
    """Check if imports in the file are valid"""
    errors = []
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        # Try to find the module
                        spec = importlib.util.find_spec(alias.name.split('.')[0])
                        if spec is None:
                            errors.append(f"Module '{alias.name}' not found")
                    except (ImportError, ModuleNotFoundError) as e:
                        errors.append(f"Module '{alias.name}': {str(e)}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        spec = importlib.util.find_spec(node.module.split('.')[0])
                        if spec is None:
                            errors.append(f"Module '{node.module}' not found")
                    except (ImportError, ModuleNotFoundError) as e:
                        errors.append(f"Module '{node.module}': {str(e)}")
        
        if errors:
            return False, "; ".join(errors)
        return True, "OK"
    except Exception as e:
        return False, f"Error checking imports: {str(e)}"

def test_file(filepath):
    """Test a single Python file"""
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print(f"{'='*60}")
    
    # Check syntax
    syntax_ok, syntax_msg = check_syntax(filepath)
    print(f"Syntax check: {'✓' if syntax_ok else '✗'} {syntax_msg}")
    
    if not syntax_ok:
        return False
    
    # Check imports
    imports_ok, imports_msg = check_imports(filepath)
    print(f"Import check: {'✓' if imports_ok else '✗'} {imports_msg}")
    
    return syntax_ok and imports_ok

def main():
    # Get all Python files in SFLLM directory
    sfllm_dir = Path(__file__).parent
    python_files = []
    
    # Add files in subdirectories
    for subdir in ['4.1-Figure3', '4.2-Table2', '4.3-Table3']:
        subdir_path = sfllm_dir / subdir
        if subdir_path.exists():
            python_files.extend(subdir_path.glob('*.py'))
    
    # Add files in main SFLLM directory
    python_files.extend(sfllm_dir.glob('*.py'))
    
    # Remove this test script
    python_files = [f for f in python_files if f.name != 'test_all.py']
    
    # Sort for consistent output
    python_files.sort()
    
    print(f"Found {len(python_files)} Python files to test")
    
    results = {}
    for filepath in python_files:
        rel_path = filepath.relative_to(sfllm_dir)
        success = test_file(filepath)
        results[str(rel_path)] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    print(f"Total files tested: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for file, success in results.items():
            if not success:
                print(f"  - {file}")
        return 1
    else:
        print("\n✓ All files passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())