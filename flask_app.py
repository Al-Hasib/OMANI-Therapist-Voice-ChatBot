from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import mimetypes
from src.agenticRAG.components.document_parsing import DocumentChunker
from src.agenticRAG.components.vectorstore import VectorStoreManager

app = Flask(__name__, template_folder='.')

# Configuration
UPLOAD_FOLDER = 'KnowledgebaseFile'
METADATA_FILE = 'knowledge_base_metadata.json'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'doc'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

def get_file_type(filename):
    """Get file type based on extension"""
    ext = filename.rsplit('.', 1)[1].lower()
    type_map = {
        'pdf': 'PDF',
        'txt': 'Text',
        'docx': 'Word Document',
        'doc': 'Word Document',
        'md': 'Markdown'
    }
    return type_map.get(ext, 'Unknown')

def load_metadata():
    """Load knowledge base metadata from JSON file"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return []

def save_metadata(metadata):
    """Save knowledge base metadata to JSON file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_kb_statistics():
    """Get real knowledge base statistics"""
    metadata = load_metadata()
    total_docs = len(metadata)
    
    total_size = 0
    last_update = None
    
    for doc in metadata:
        filepath = os.path.join(UPLOAD_FOLDER, doc['filename'])
        if os.path.exists(filepath):
            total_size += get_file_size(filepath)
            doc_date = datetime.fromisoformat(doc['uploaded'])
            if last_update is None or doc_date > last_update:
                last_update = doc_date
    
    return {
        'total_docs': total_docs,
        'total_size': round(total_size, 2),
        'last_update': last_update.strftime('%Y-%m-%d') if last_update else 'Never'
    }

def get_knowledge_base():
    """Get current knowledge base with real file information"""
    metadata = load_metadata()
    kb_data = []
    
    for doc in metadata:
        filepath = os.path.join(UPLOAD_FOLDER, doc['filename'])
        if os.path.exists(filepath):
            kb_data.append({
                'id': doc['id'],
                'name': doc['filename'],
                'size': f"{get_file_size(filepath)} MB",
                'type': get_file_type(doc['filename']),
                'uploaded': doc['uploaded'],
                'chunks': doc.get('chunks', 'N/A'),
                'description': doc['description']
            })
    
    return kb_data

@app.route('/')
def index():
    return render_template('knowledgebase_UI.html')

@app.route('/api/statistics')
def get_statistics():
    """API endpoint to get knowledge base statistics"""
    return jsonify(get_kb_statistics())

@app.route('/api/knowledge-base')
def get_kb_data():
    """API endpoint to get knowledge base data"""
    return jsonify(get_knowledge_base())

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file upload with description"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        description = request.form.get('description', '').strip()
        
        if not description:
            return jsonify({'error': 'Description is required'}), 400
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        metadata = load_metadata()
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Handle duplicate filenames
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
                    filename = f"{base_name}_{counter}{ext}"
                    counter += 1
                
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                print(f"File saved: {filepath}")
                # Create DocumentChunker instance
                chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
                # Process the file to create chunks
                chunks = chunker.process_file(filepath)
                print(f"Chunks created: {len(chunks)} for {filename}")

                # Add to vector store
                vector_store_manager = VectorStoreManager()
                vector_store_manager.load_vectorstore()
                
                # Create metadata for each chunk - THIS IS THE FIX
                chunk_metadatas = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        'filename': filename,
                        'description': description,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'uploaded': datetime.now().isoformat()
                    }
                    chunk_metadatas.append(chunk_metadata)
                
                # Now texts and metadatas have the same length
                vector_store_manager.add_documents(chunks, metadatas=chunk_metadatas)
                vector_store_manager.save_vectorstore()

                # Create metadata entry for the file
                doc_metadata = {
                    'id': len(metadata) + 1,
                    'filename': filename,
                    'description': description,
                    'uploaded': datetime.now().isoformat(),
                    'chunks': len(chunks) if chunks else 0,
                }
                
                metadata.append(doc_metadata)
                uploaded_files.append(filename)
        
        # Save updated metadata
        save_metadata(metadata)
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/delete/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from knowledge base"""
    try:
        metadata = load_metadata()
        doc_to_delete = None
        
        for i, doc in enumerate(metadata):
            if doc['id'] == doc_id:
                doc_to_delete = doc
                metadata.pop(i)
                break
        
        if not doc_to_delete:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete the actual file
        filepath = os.path.join(UPLOAD_FOLDER, doc_to_delete['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Save updated metadata
        save_metadata(metadata)
        
        return jsonify({'message': f'Successfully deleted {doc_to_delete["filename"]}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-all', methods=['DELETE'])
def clear_knowledge_base():
    """Clear entire knowledge base"""
    try:
        metadata = load_metadata()
        
        # Delete all files
        for doc in metadata:
            filepath = os.path.join(UPLOAD_FOLDER, doc['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Clear metadata
        save_metadata([])
        
        return jsonify({'message': 'Knowledge base cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/<int:doc_id>')
def get_document_details(doc_id):
    """Get detailed information about a specific document"""
    try:
        metadata = load_metadata()
        
        for doc in metadata:
            if doc['id'] == doc_id:
                filepath = os.path.join(UPLOAD_FOLDER, doc['filename'])
                if os.path.exists(filepath):
                    return jsonify({
                        'id': doc['id'],
                        'name': doc['filename'],
                        'description': doc['description'],
                        'size': f"{get_file_size(filepath)} MB",
                        'type': get_file_type(doc['filename']),
                        'uploaded': doc['uploaded'],
                        'chunks': doc.get('chunks', 'N/A'),
                        'path': filepath
                    })
                else:
                    return jsonify({'error': 'File not found on disk'}), 404
        
        return jsonify({'error': 'Document not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search_documents():
    """Search documents by name or description"""
    try:
        query = request.args.get('q', '').lower()
        
        if not query:
            return jsonify(get_knowledge_base())
        
        metadata = load_metadata()
        results = []
        
        for doc in metadata:
            if (query in doc['filename'].lower() or 
                query in doc['description'].lower()):
                
                filepath = os.path.join(UPLOAD_FOLDER, doc['filename'])
                if os.path.exists(filepath):
                    results.append({
                        'id': doc['id'],
                        'name': doc['filename'],
                        'size': f"{get_file_size(filepath)} MB",
                        'type': get_file_type(doc['filename']),
                        'uploaded': doc['uploaded'],
                        'chunks': doc.get('chunks', 'N/A'),
                        'description': doc['description']
                    })
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Knowledge Base files will be stored in: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Metadata will be stored in: {os.path.abspath(METADATA_FILE)}")
    app.run(debug=True, host='0.0.0.0', port=5001)