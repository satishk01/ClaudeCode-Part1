import streamlit as st
import boto3
import json
import os
import re
from datetime import datetime
import base64
import uuid
from pathlib import Path
import time
import logging
from botocore.config import Config

# Set page configuration
st.set_page_config(
    page_title="AWS Bedrock Code Generator",
    page_icon="ðŸ§ ",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'folder_name' not in st.session_state:
    st.session_state.folder_name = ""

# Function to read the system prompt file
def load_system_prompt():
    with open("advaned-system-prompts.txt", "r") as file:
        return file.read()

# Function to load example prompts
def load_example_prompts():
    with open("advanced-example-prompts.txt", "r") as file:
        return file.read()

# Function to generate prompt format based on examples and requirements
def format_prompt(requirement):
    # This is a simple template based on the examples provided
    prompt = f"""
Based on the following requirement, please generate complete, production-ready Node.js Express API code:

REQUIREMENT:
{requirement}

Please follow all the architecture patterns, coding standards, and implementation guidelines as specified in the system prompt.
Ensure the code is:
- Well-structured with clear separation of concerns
- Thoroughly documented with JSDoc comments
- Consistent with established patterns
- Optimized for performance and reliability
- Free from security vulnerabilities
- Ready for production use without significant modifications

Please provide the complete code implementation.
"""
    return prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bedrock_api.log')
    ]
)
logger = logging.getLogger('bedrock_api')

# Function to generate code in stages to avoid timeout issues
def generate_code_in_stages(system_prompt, user_prompt):
    logger.info("Starting staged code generation process")
    
    # Stage 1: Generate initial code scaffold with basic structure
    stage1_prompt = f"{user_prompt}\n\nStart by generating the overall structure of the Node.js Express API with the main file organization and essential imports."
    stage1_result = invoke_claude_model(system_prompt, stage1_prompt)
    
    # Log progress and update UI
    logger.info("Stage 1 complete: Code scaffold generated")
    st.text("✓ Generated API structure and scaffolding")
    
    # Stage 2: Generate model definitions and utility functions
    stage2_prompt = f"{user_prompt}\n\nBased on the following initial code structure, please expand it with detailed model definitions and utility functions:\n\n{stage1_result}"
    stage2_result = invoke_claude_model(system_prompt, stage2_prompt)
    
    # Log progress and update UI
    logger.info("Stage 2 complete: Models and utilities generated")
    st.text("✓ Generated models and utility functions")
    
    # Stage 3: Generate route handlers and OpenSearch integration
    stage3_prompt = f"{user_prompt}\n\nBased on the following code with models and utilities, please complete the implementation with detailed route handlers and OpenSearch integration:\n\n{stage2_result}"
    final_result = invoke_claude_model(system_prompt, stage3_prompt)
    
    # Log completion and return final code
    logger.info("Stage 3 complete: Full code generation finished")
    st.text("✓ Generated routes and OpenSearch integration")
    
    return final_result

# Function to invoke Claude model on AWS Bedrock with retry mechanism
def invoke_claude_model(system_prompt, user_prompt, max_retries=3):
    # Configure the AWS SDK with custom timeouts
    config = Config(
        region_name='us-east-1',
        connect_timeout=120,  # 2 minutes connection timeout
        read_timeout=300,     # 5 minutes read timeout
        retries={'max_attempts': 0}  # We'll handle retries manually
    )
    
    bedrock_runtime = boto3.client('bedrock-runtime', config=config)
    
    # Initial setup for token chunking
    max_tokens_per_request = 2000  # Smaller chunks to avoid timeouts
    
    # Prepare the base request body
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens_per_request,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    # Retry logic
    retries = 0
    full_response = ""
    
    while retries <= max_retries:
        try:
            logger.info(f"Making Bedrock API call (attempt {retries+1}/{max_retries+1})")
            start_time = time.time()
            
            # Make the API call
            response = bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps(request_body)
            )
            
            # Process response
            response_body = json.loads(response.get('body').read())
            response_text = response_body.get('content')[0].get('text')
            full_response += response_text
            
            # Log successful completion
            elapsed_time = time.time() - start_time
            logger.info(f"API call successful. Elapsed time: {elapsed_time:.2f} seconds")
            
            # If we need more content, update the prompt for the next request
            if len(response_text) >= max_tokens_per_request * 0.9:  # If we're getting close to the limit
                # For staged generation, we would update the user_prompt here
                # But for this initial implementation, we'll return what we have
                logger.info("Response truncated due to token limit")
                break
            
            # If we got a complete response, break out of the retry loop
            return full_response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"API call failed after {elapsed_time:.2f} seconds: {str(e)}")
            retries += 1
            
            if retries <= max_retries:
                # Exponential backoff
                wait_time = 2 ** retries
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Maximum retries ({max_retries}) exceeded")
                raise
    
    return full_response

# Function to save generated code to file
def save_code_to_file(code, folder_name, file_name):
    # Create folder if it doesn't exist
    folder_path = Path("generatedcode") / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Check if the code contains references to additional files
    files_to_create = extract_file_references(code)
    
    # If we have structured files, save them individually
    if files_to_create and len(files_to_create) > 0:
        logger.info(f"Found {len(files_to_create)} structured files to create")
        
        for file_info in files_to_create:
            # Create subdirectories if needed
            file_path_parts = file_info['path'].split('/')
            filename = file_path_parts[-1]
            
            if len(file_path_parts) > 1:
                # There are subdirectories in the path
                subdir = '/'.join(file_path_parts[:-1])
                file_dir = folder_path / subdir
                file_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            sub_file_path = folder_path / file_info['path']
            with open(sub_file_path, "w") as file:
                file.write(file_info['content'])
            
            logger.info(f"Created file: {sub_file_path}")
            
        # Use the first file as our main file for display purposes
        main_file_path = folder_path / files_to_create[0]['path']
        # Update the code for display in the UI
        code = files_to_create[0]['content']
        
        return str(main_file_path)
    else:
        # Save as a single file if no structured files were detected
        logger.info("No structured files found, saving as a single file")
        file_path = folder_path / file_name
        with open(file_path, "w") as file:
            file.write(code)
        
        return str(file_path)

# Function to extract file references and content from generated code
def extract_file_references(code):
    files = []
    lines = code.split('\n')
    
    # Process multiple file blocks in the response
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Pattern 1: Explicit file path headers like '**src/app.js**' or 'src/app.js:' or '# src/app.js'
        file_header_match = re.search(r'\*\*([\w\-./]+\.\w+)\*\*|^([\w\-./]+\.\w+):$|^#\s+([\w\-./]+\.\w+)$', line)
        
        # Pattern 2: Numbered file references like '1. src/models/user.js'
        numbered_file_match = re.search(r'^\d+\.\s+([\w\-./]+\.\w+)\s*$', line)
        
        # Pattern 3: Commented file references like '// filename: src/config/db.js'
        comment_file_match = re.search(r'//\s*filename:\s*([\w\-./]+\.\w+)\s*$', line)
        
        # Pattern 4: File path before code block like 'src/app.js' followed by ```
        if i + 1 < len(lines) and (lines[i+1].strip().startswith('```')):
            file_before_block_match = re.search(r'^([\w\-./]+\.\w+)\s*$', line)
        else:
            file_before_block_match = None
        
        # If any file path pattern matched
        if file_header_match or numbered_file_match or comment_file_match or file_before_block_match:
            # Extract the file path from whichever match succeeded
            if file_header_match:
                # Get the first non-None group
                file_path = next(g for g in file_header_match.groups() if g is not None)
            elif numbered_file_match:
                file_path = numbered_file_match.group(1)
            elif comment_file_match:
                file_path = comment_file_match.group(1)
            elif file_before_block_match:
                file_path = file_before_block_match.group(1)
            
            # Move to next line to find code block
            j = i + 1
            # Skip any explanatory text until we find a code block
            while j < len(lines) and not lines[j].strip().startswith('```'):
                j += 1
            
            # If we found a code block
            if j < len(lines) and lines[j].strip().startswith('```'):
                j += 1  # Skip the opening ```
                code_content = []
                
                # Collect all content until closing ```
                while j < len(lines) and not lines[j].strip() == '```':
                    code_content.append(lines[j])
                    j += 1
                
                # Add file with its content
                files.append({
                    'path': file_path,
                    'content': '\n'.join(code_content)
                })
                
                # Move outer loop index to after this code block
                if j < len(lines):
                    i = j + 1
                else:
                    i = j
                continue
        
        # Pattern 5: Code blocks with file path embedded in the language specifier ```js:src/app.js
        code_block_with_path_match = re.search(r'^```\w*:([\w\-./]+\.\w+)$', line)
        if code_block_with_path_match:
            file_path = code_block_with_path_match.group(1)
            
            # Extract content between code blocks
            j = i + 1
            code_content = []
            while j < len(lines) and not lines[j].strip() == '```':
                code_content.append(lines[j])
                j += 1
            
            # Add file with its content
            files.append({
                'path': file_path,
                'content': '\n'.join(code_content)
            })
            
            # Move outer loop index to after this code block
            if j < len(lines):
                i = j + 1
            else:
                i = j
            continue
        
        i += 1
    
    # Special case: If we only got a single block of code with no filename, assume it's app.js
    if len(files) == 0 and '```' in code:
        code_parts = code.split('```')
        if len(code_parts) >= 3:
            code_content = code_parts[1]
            # Remove language identifier if present (e.g., "js\n")
            if "\n" in code_content:
                parts = code_content.split("\n", 1)
                if len(parts) > 1:
                    code_content = parts[1]
            files.append({
                'path': 'app.js',  # Default filename
                'content': code_content
            })
    
    # Make sure we have at least some files
    # If there's actual code but no files detected, create a default file structure
    if len(files) == 0 and len(code) > 100 and not all(f['path'] for f in files):
        files.append({
            'path': 'app.js',  # Default main file
            'content': code.strip()
        })
    
    return files

# Function to create a download link for the generated code
def get_binary_file_downloader_html(file_path, file_name):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Main UI layout
st.title("AWS Bedrock Code Generator")
st.subheader("Powered by Anthropic Claude 3.5 Sonnet")

with st.expander("About this app"):
    st.write("""
    This application uses AWS Bedrock's Anthropic Claude 3.5 Sonnet model to generate Node.js Express API code
    based on user requirements. The system is designed to follow specific architecture patterns and coding standards
    for OpenSearch integrations.
    
    1. Enter your code generation requirement
    2. The app will format your requirement as a prompt
    3. AWS Bedrock will generate code based on the system prompt and your requirement
    4. The generated code will be displayed and saved
    5. You can download the code or find it in the generatedcode folder
    """)

# Input section
st.header("Enter Your Requirement")
user_requirement = st.text_area(
    "Describe the code you need",
    height=150,
    placeholder="Example: Create a Node.js Express API that connects to OpenSearch for querying vehicle inventory data..."
)

# Settings section
st.subheader("Settings")
col1, col2 = st.columns(2)

with col1:
    folder_name = st.text_input(
        "Folder Name", 
        value=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

with col2:
    file_name = st.text_input(
        "File Name", 
        value="index.js"
    )

# Action button
if st.button("Generate Code"):
    if user_requirement:
        with st.spinner("Generating code... This may take a minute or two..."):
            try:
                # Load system prompt
                system_prompt = load_system_prompt()
                
                # Format user prompt
                user_prompt = format_prompt(user_requirement)
                
                # Generate code in stages to avoid timeouts
                st.text("Generating code in stages to avoid timeouts...")
                generated_code = generate_code_in_stages(system_prompt, user_prompt)
                
                # Extract code from potential markdown code blocks in the response
                if "```" in generated_code:
                    # Extract code between the first and last code block markers
                    code_parts = generated_code.split("```")
                    if len(code_parts) >= 3:
                        # If the format is ```js\ncode\n```
                        # code_parts[0] is text before the first marker
                        # code_parts[1] might be "js\ncode" or just "code"
                        code_content = code_parts[1]
                        # Remove language identifier if present (e.g., "js\n")
                        if "\n" in code_content:
                            # Split by first newline
                            parts = code_content.split("\n", 1)
                            if len(parts) > 1:
                                code_content = parts[1]
                        st.session_state.generated_code = code_content
                    else:
                        st.session_state.generated_code = generated_code
                else:
                    st.session_state.generated_code = generated_code
                
                st.session_state.file_name = file_name
                st.session_state.folder_name = folder_name
                
                # Save to file
                file_path = save_code_to_file(
                    st.session_state.generated_code, 
                    st.session_state.folder_name, 
                    st.session_state.file_name
                )
                
                st.success(f"Code generated and saved to {file_path}")
            
            except Exception as e:
                st.error(f"Error during code generation: {str(e)}")
                logger.error(f"Code generation failed: {str(e)}", exc_info=True)
                st.error("Please check your AWS Bedrock permissions or try again later.")
    else:
        st.warning("Please enter a requirement before generating code.")

# Display generated code
if st.session_state.generated_code:
    st.header("Generated Code")
    
    # Display file and folder path information
    st.subheader("Generated Code Location")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Folder path:** `generatedcode/{st.session_state.folder_name}/`")
    with col2:
        st.info(f"**File path:** `generatedcode/{st.session_state.folder_name}/{st.session_state.file_name}`")
    
    # Display file structure
    if st.session_state.folder_name:
        folder_path = Path("generatedcode") / st.session_state.folder_name
        if folder_path.exists():
            st.subheader("Project Structure")
            all_files = []
            for root, dirs, files in os.walk(folder_path):
                rel_path = os.path.relpath(root, start=Path("generatedcode"))
                for file in files:
                    all_files.append(os.path.join(rel_path, file))
            
            if all_files:
                st.code("\n".join(sorted(all_files)), language="bash")
            else:
                st.info("No files found in project directory.")
    
    # Show code content
    st.subheader(f"Content of {st.session_state.file_name}")
    st.code(st.session_state.generated_code, language='javascript')
    
    # Download link
    if st.session_state.file_name and st.session_state.folder_name:
        file_path = f"generatedcode/{st.session_state.folder_name}/{st.session_state.file_name}"
        
        # Create download button
        with open(file_path, "rb") as f:
            st.download_button(
                label="Download Generated Code",
                data=f.read(),
                file_name=st.session_state.file_name,
                mime="text/plain"
            )