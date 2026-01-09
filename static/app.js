document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const topKInput = document.getElementById('topKInput');
    const resultsGrid = document.getElementById('resultsGrid');
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');

    // Modal elements
    const indexBtn = document.getElementById('indexBtn');
    const indexModal = document.getElementById('indexModal');
    const closeBtn = document.querySelector('.close-btn');
    const confirmIndexBtn = document.getElementById('confirmIndexBtn');
    const folderPathInput = document.getElementById('folderPathInput');

    // Progress elements
    const progressContainer = document.getElementById('progressContainer');
    const progressText = document.getElementById('progressText');
    const progressPercent = document.getElementById('progressPercent');
    const progressBarFill = document.getElementById('progressBarFill');
    const progressDetail = document.getElementById('progressDetail');

    // Search Functionality
    async function performSearch() {
        const query = searchInput.value.trim();
        const topK = parseInt(topKInput.value) || 20;

        if (!query) return;

        // UI State
        emptyState.classList.add('hidden');
        resultsGrid.innerHTML = '';
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, top_k: topK }),
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error('Error:', error);
            resultsGrid.innerHTML = `<div style="text-align: center; color: red;">Error performing search: ${error.message}</div>`;
        } finally {
            loadingState.classList.add('hidden');
        }
    }

    function renderResults(data) {
        resultsGrid.innerHTML = '';

        let allResults = [];

        // Prefer GTE, then others
        if (data.gte) allResults = [...data.gte];
        else if (data.gemma) allResults = [...data.gemma];
        else if (data.qwen) allResults = [...data.qwen];

        if (allResults.length === 0) {
            resultsGrid.innerHTML = '<p style="text-align: center; width: 100%; color: #666;">No matches found.</p>';
            return;
        }

        allResults.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 50}ms`;

            // Construct safe image URL
            const imageUrl = `/api/images?path=${encodeURIComponent(item.path)}`;

            card.innerHTML = `
                <div class="image-container">
                    <img src="${imageUrl}" alt="${item.caption}" loading="lazy">
                </div>
                <div class="card-info">
                    <span class="similarity-badge">${(item.similarity * 100).toFixed(1)}%</span>
                    <p class="caption">${item.caption}</p>
                </div>
            `;

            // Add click handler to open full image
            card.addEventListener('click', () => {
                window.open(imageUrl, '_blank');
            });

            resultsGrid.appendChild(card);
        });
    }

    // Indexing Functionality with Streaming Progress
    async function performIndexing() {
        const path = folderPathInput.value.trim();
        if (!path) return;

        confirmIndexBtn.textContent = 'Indexing...';
        confirmIndexBtn.disabled = true;

        // Reset and show progress
        progressContainer.classList.remove('hidden');
        progressBarFill.style.width = '0%';
        progressPercent.textContent = '0%';
        progressText.textContent = 'Starting...';
        progressDetail.textContent = '';

        try {
            const response = await fetch('/api/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            // Read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                // Process all complete lines
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const status = JSON.parse(line);
                        updateProgressUI(status);
                    } catch (e) {
                        console.error('Error parsing JSON line:', e);
                    }
                }
            }

        } catch (error) {
            alert(`Error: ${error.message}`);
            progressText.textContent = 'Error';
            progressDetail.textContent = error.message;
        } finally {
            confirmIndexBtn.textContent = 'Start Indexing';
            confirmIndexBtn.disabled = false;
        }
    }

    function updateProgressUI(status) {
        if (status.status === 'info' || status.status === 'start') {
            progressText.textContent = status.message || 'Initializing...';
        }
        else if (status.status === 'progress') {
            const percent = Math.round((status.current / status.total) * 100);
            progressBarFill.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
            progressText.textContent = `Processing ${status.current}/${status.total}`;
            // Only show filename, not caption as requested
            progressDetail.textContent = status.file;
        }
        else if (status.status === 'embedding') {
            progressBarFill.style.width = '100%';
            progressPercent.textContent = '100%';
            progressText.textContent = 'Generating Embeddings...';
            progressDetail.textContent = 'This may take a moment';
        }
        else if (status.status === 'complete') {
            progressBarFill.style.width = '100%';
            progressPercent.textContent = 'Done';
            progressText.textContent = 'Indexing Complete!';
            progressDetail.textContent = status.message;

            // Auto close after delay
            setTimeout(() => {
                // indexModal.classList.add('hidden');
                alert(`Indexing Complete! Added ${status.indexed_count || 0} images.`);
            }, 500);
        }
        else if (status.status === 'error') {
            progressText.textContent = 'Error';
            progressDetail.textContent = status.message;
            progressBarFill.style.backgroundColor = '#ef4444';
        }
    }

    // Event Listeners
    searchBtn.addEventListener('click', performSearch);

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    indexBtn.addEventListener('click', () => {
        indexModal.classList.remove('hidden');
        folderPathInput.focus();
        progressContainer.classList.add('hidden'); // Hide progress on new open
        progressBarFill.style.width = '0%';
    });

    closeBtn.addEventListener('click', () => {
        indexModal.classList.add('hidden');
    });

    // Close modal on click outside
    indexModal.addEventListener('click', (e) => {
        if (e.target === indexModal) {
            indexModal.classList.add('hidden');
        }
    });

    confirmIndexBtn.addEventListener('click', performIndexing);
});
