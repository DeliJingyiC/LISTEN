// Leaderboard data from paper results
// Table shows: Neutral-Text (Exp1), Emotion-Matched (Exp2), Emotion-Mismatched (Exp3), Paralinguistic (Exp4)
// Values in parentheses are prediction marginal distribution baselines
const leaderboardData = [
    {
        model: "Gemini 2.5 Pro",
        type: "proprietary",
        exp1_text: { accuracy: 96.6, weighted_accuracy: 96.6, uar: 96.6, macro_f1: 96.6, micro_f1: 96.6, baseline: 96.6 },
        exp1_audio: { accuracy: 34.9, weighted_accuracy: 34.9, uar: 34.9, macro_f1: 34.9, micro_f1: 34.9, baseline: 12.8 },
        exp1_both: { accuracy: 41.7, weighted_accuracy: 41.7, uar: 41.7, macro_f1: 41.7, micro_f1: 41.7, baseline: 12.9 },
        exp2_text: { accuracy: 38.8, weighted_accuracy: 38.8, uar: 38.8, macro_f1: 38.8, micro_f1: 38.8, baseline: 15.7 },
        exp2_audio: { accuracy: 37.6, weighted_accuracy: 37.6, uar: 37.6, macro_f1: 37.6, micro_f1: 37.6, baseline: 16.9 },
        exp2_both: { accuracy: 40.2, weighted_accuracy: 40.2, uar: 40.2, macro_f1: 40.2, micro_f1: 40.2, baseline: 14.3 },
        exp3_text: { accuracy: 31.8, weighted_accuracy: 31.8, uar: 31.8, macro_f1: 31.8, micro_f1: 31.8, baseline: 10.0 },
        exp3_audio: { accuracy: 36.9, weighted_accuracy: 36.9, uar: 36.9, macro_f1: 36.9, micro_f1: 36.9, baseline: 22.5 },
        exp3_both: { accuracy: 42.6, weighted_accuracy: 42.6, uar: 42.6, macro_f1: 42.6, micro_f1: 42.6, baseline: 23.3 },
        exp4_audio: { accuracy: 15.7, weighted_accuracy: 15.7, uar: 15.7, macro_f1: 15.7, micro_f1: 15.7, baseline: 9.3 }
    },
    {
        model: "Qwen3-Omni-30B",
        type: "opensource",
        exp1_text: { accuracy: 85.4, weighted_accuracy: 85.4, uar: 85.4, macro_f1: 85.4, micro_f1: 85.4, baseline: 84.0 },
        exp1_audio: { accuracy: 29.3, weighted_accuracy: 29.3, uar: 29.3, macro_f1: 29.3, micro_f1: 29.3, baseline: 11.3 },
        exp1_both: { accuracy: 25.3, weighted_accuracy: 25.3, uar: 25.3, macro_f1: 25.3, micro_f1: 25.3, baseline: 11.8 },
        exp2_text: { accuracy: 38.7, weighted_accuracy: 38.7, uar: 38.7, macro_f1: 38.7, micro_f1: 38.7, baseline: 11.7 },
        exp2_audio: { accuracy: 42.4, weighted_accuracy: 42.4, uar: 42.4, macro_f1: 42.4, micro_f1: 42.4, baseline: 15.8 },
        exp2_both: { accuracy: 43.1, weighted_accuracy: 43.1, uar: 43.1, macro_f1: 43.1, micro_f1: 43.1, baseline: 15.5 },
        exp3_text: { accuracy: 34.6, weighted_accuracy: 34.6, uar: 34.6, macro_f1: 34.6, micro_f1: 34.6, baseline: 12.2 },
        exp3_audio: { accuracy: 37.4, weighted_accuracy: 37.4, uar: 37.4, macro_f1: 37.4, micro_f1: 37.4, baseline: 26.7 },
        exp3_both: { accuracy: 39.1, weighted_accuracy: 39.1, uar: 39.1, macro_f1: 39.1, micro_f1: 39.1, baseline: 25.2 },
        exp4_audio: { accuracy: 21.0, weighted_accuracy: 21.0, uar: 21.0, macro_f1: 21.0, micro_f1: 21.0, baseline: 12.6 }
    },
    {
        model: "Qwen2.5-Omni-7B",
        type: "opensource",
        exp1_text: { accuracy: 85.4, weighted_accuracy: 85.4, uar: 85.4, macro_f1: 85.4, micro_f1: 85.4, baseline: 84.1 },
        exp1_audio: { accuracy: 34.0, weighted_accuracy: 34.0, uar: 34.0, macro_f1: 34.0, micro_f1: 34.0, baseline: 10.9 },
        exp1_both: { accuracy: 19.8, weighted_accuracy: 19.8, uar: 19.8, macro_f1: 19.8, micro_f1: 19.8, baseline: 11.8 },
        exp2_text: { accuracy: 36.4, weighted_accuracy: 36.4, uar: 36.4, macro_f1: 36.4, micro_f1: 36.4, baseline: 15.7 },
        exp2_audio: { accuracy: 36.6, weighted_accuracy: 36.6, uar: 36.6, macro_f1: 36.6, micro_f1: 36.6, baseline: 14.1 },
        exp2_both: { accuracy: 38.6, weighted_accuracy: 38.6, uar: 38.6, macro_f1: 38.6, micro_f1: 38.6, baseline: 15.7 },
        exp3_text: { accuracy: 34.0, weighted_accuracy: 34.0, uar: 34.0, macro_f1: 34.0, micro_f1: 34.0, baseline: 13.5 },
        exp3_audio: { accuracy: 38.5, weighted_accuracy: 38.5, uar: 38.5, macro_f1: 38.5, micro_f1: 38.5, baseline: 29.5 },
        exp3_both: { accuracy: 39.1, weighted_accuracy: 39.1, uar: 39.1, macro_f1: 39.1, micro_f1: 39.1, baseline: 23.6 },
        exp4_audio: { accuracy: 22.7, weighted_accuracy: 22.7, uar: 22.7, macro_f1: 22.7, micro_f1: 22.7, baseline: 11.4 }
    },
    {
        model: "Gemini 2.5 Flash",
        type: "proprietary",
        exp1_text: { accuracy: 82.5, weighted_accuracy: 82.5, uar: 82.5, macro_f1: 82.5, micro_f1: 82.5, baseline: 81.5 },
        exp1_audio: { accuracy: 25.6, weighted_accuracy: 25.6, uar: 25.6, macro_f1: 25.6, micro_f1: 25.6, baseline: 10.7 },
        exp1_both: { accuracy: 24.6, weighted_accuracy: 24.6, uar: 24.6, macro_f1: 24.6, micro_f1: 24.6, baseline: 11.0 },
        exp2_text: { accuracy: 36.6, weighted_accuracy: 36.6, uar: 36.6, macro_f1: 36.6, micro_f1: 36.6, baseline: 14.3 },
        exp2_audio: { accuracy: 30.7, weighted_accuracy: 30.7, uar: 30.7, macro_f1: 30.7, micro_f1: 30.7, baseline: 12.3 },
        exp2_both: { accuracy: 38.9, weighted_accuracy: 38.9, uar: 38.9, macro_f1: 38.9, micro_f1: 38.9, baseline: 14.1 },
        exp3_text: { accuracy: 33.1, weighted_accuracy: 33.1, uar: 33.1, macro_f1: 33.1, micro_f1: 33.1, baseline: 10.1 },
        exp3_audio: { accuracy: 35.8, weighted_accuracy: 35.8, uar: 35.8, macro_f1: 35.8, micro_f1: 35.8, baseline: 15.5 },
        exp3_both: { accuracy: 38.0, weighted_accuracy: 38.0, uar: 38.0, macro_f1: 38.0, micro_f1: 38.0, baseline: 18.5 },
        exp4_audio: { accuracy: 18.0, weighted_accuracy: 18.0, uar: 18.0, macro_f1: 18.0, micro_f1: 18.0, baseline: 12.0 }
    },
    {
        model: "Baichuan-Omni-1.5",
        type: "opensource",
        exp1_text: { accuracy: 81.2, weighted_accuracy: 81.2, uar: 81.2, macro_f1: 81.2, micro_f1: 81.2, baseline: 79.6 },
        exp1_audio: { accuracy: 16.5, weighted_accuracy: 16.5, uar: 16.5, macro_f1: 16.5, micro_f1: 16.5, baseline: 11.5 },
        exp1_both: { accuracy: 15.2, weighted_accuracy: 15.2, uar: 15.2, macro_f1: 15.2, micro_f1: 15.2, baseline: 12.1 },
        exp2_text: { accuracy: 31.0, weighted_accuracy: 31.0, uar: 31.0, macro_f1: 31.0, micro_f1: 31.0, baseline: 15.9 },
        exp2_audio: { accuracy: 36.0, weighted_accuracy: 36.0, uar: 36.0, macro_f1: 36.0, micro_f1: 36.0, baseline: 15.6 },
        exp2_both: { accuracy: 36.0, weighted_accuracy: 36.0, uar: 36.0, macro_f1: 36.0, micro_f1: 36.0, baseline: 17.8 },
        exp3_text: { accuracy: 31.0, weighted_accuracy: 31.0, uar: 31.0, macro_f1: 31.0, micro_f1: 31.0, baseline: 27.5 },
        exp3_audio: { accuracy: 36.0, weighted_accuracy: 36.0, uar: 36.0, macro_f1: 36.0, micro_f1: 36.0, baseline: 31.0 },
        exp3_both: { accuracy: 36.0, weighted_accuracy: 36.0, uar: 36.0, macro_f1: 36.0, micro_f1: 36.0, baseline: 32.0 },
        exp4_audio: { accuracy: 22.7, weighted_accuracy: 22.7, uar: 22.7, macro_f1: 22.7, micro_f1: 22.7, baseline: 11.5 }
    },
    {
        model: "Qwen3-Instruct",
        type: "opensource",
        exp1_text: { accuracy: 66.6, weighted_accuracy: 66.6, uar: 66.6, macro_f1: 66.6, micro_f1: 66.6, baseline: 65.0 },
        exp1_audio: null,
        exp1_both: null,
        exp2_text: { accuracy: 33.5, weighted_accuracy: 33.5, uar: 33.5, macro_f1: 33.5, micro_f1: 33.5, baseline: 12.4 },
        exp2_audio: null,
        exp2_both: null,
        exp3_text: { accuracy: 38.0, weighted_accuracy: 38.0, uar: 38.0, macro_f1: 38.0, micro_f1: 38.0, baseline: 29.8 },
        exp3_audio: null,
        exp3_both: null,
        exp4_audio: null
    },
    {
        model: "Uniform Guess",
        type: "baseline",
        exp1_text: { accuracy: 12.5, weighted_accuracy: 12.5, uar: 12.5, macro_f1: 12.5, micro_f1: 12.5, baseline: 12.5 },
        exp1_audio: { accuracy: 12.5, weighted_accuracy: 12.5, uar: 12.5, macro_f1: 12.5, micro_f1: 12.5, baseline: 12.5 },
        exp1_both: { accuracy: 12.5, weighted_accuracy: 12.5, uar: 12.5, macro_f1: 12.5, micro_f1: 12.5, baseline: 12.5 },
        exp2_text: { accuracy: 6.7, weighted_accuracy: 6.7, uar: 6.7, macro_f1: 6.7, micro_f1: 6.7, baseline: 6.7 },
        exp2_audio: { accuracy: 6.7, weighted_accuracy: 6.7, uar: 6.7, macro_f1: 6.7, micro_f1: 6.7, baseline: 6.7 },
        exp2_both: { accuracy: 6.7, weighted_accuracy: 6.7, uar: 6.7, macro_f1: 6.7, micro_f1: 6.7, baseline: 6.7 },
        exp3_text: { accuracy: 10.0, weighted_accuracy: 10.0, uar: 10.0, macro_f1: 10.0, micro_f1: 10.0, baseline: 10.0 },
        exp3_audio: { accuracy: 10.0, weighted_accuracy: 10.0, uar: 10.0, macro_f1: 10.0, micro_f1: 10.0, baseline: 10.0 },
        exp3_both: { accuracy: 10.0, weighted_accuracy: 10.0, uar: 10.0, macro_f1: 10.0, micro_f1: 10.0, baseline: 10.0 },
        exp4_audio: { accuracy: 12.5, weighted_accuracy: 12.5, uar: 12.5, macro_f1: 12.5, micro_f1: 12.5, baseline: 12.5 }
    },
    {
        model: "Majority Guess",
        type: "baseline",
        exp1_text: { accuracy: 100.0, weighted_accuracy: 100.0, uar: 100.0, macro_f1: 100.0, micro_f1: 100.0, baseline: 100.0 },
        exp1_audio: { accuracy: 16.9, weighted_accuracy: 16.9, uar: 16.9, macro_f1: 16.9, micro_f1: 16.9, baseline: 16.9 },
        exp1_both: { accuracy: 16.9, weighted_accuracy: 16.9, uar: 16.9, macro_f1: 16.9, micro_f1: 16.9, baseline: 16.9 },
        exp2_text: { accuracy: 26.5, weighted_accuracy: 26.5, uar: 26.5, macro_f1: 26.5, micro_f1: 26.5, baseline: 26.5 },
        exp2_audio: { accuracy: 26.5, weighted_accuracy: 26.5, uar: 26.5, macro_f1: 26.5, micro_f1: 26.5, baseline: 26.5 },
        exp2_both: { accuracy: 26.5, weighted_accuracy: 26.5, uar: 26.5, macro_f1: 26.5, micro_f1: 26.5, baseline: 26.5 },
        exp3_text: { accuracy: 39.0, weighted_accuracy: 39.0, uar: 39.0, macro_f1: 39.0, micro_f1: 39.0, baseline: 39.0 },
        exp3_audio: { accuracy: 39.0, weighted_accuracy: 39.0, uar: 39.0, macro_f1: 39.0, micro_f1: 39.0, baseline: 39.0 },
        exp3_both: { accuracy: 39.0, weighted_accuracy: 39.0, uar: 39.0, macro_f1: 39.0, micro_f1: 39.0, baseline: 39.0 },
        exp4_audio: { accuracy: 32.6, weighted_accuracy: 32.6, uar: 32.6, macro_f1: 32.6, micro_f1: 32.6, baseline: 32.6 }
    }
];

// State management
let currentSort = { column: 'avg', ascending: false };
let currentMetric = 'accuracy';
let currentExperiment = 'all';

// Calculate average score for a model (Overall Average from paper)
// Only includes audio and text+audio results (excludes text-only)
function calculateAverage(model, metric) {
    // Overall Average = mean of audio and text+audio across all 4 conditions (7 modalities total)
    const experiments = ['exp1_audio', 'exp1_both',
        'exp2_audio', 'exp2_both',
        'exp3_audio', 'exp3_both',
        'exp4_audio'];

    let sum = 0;
    let count = 0;

    experiments.forEach(exp => {
        if (model[exp] && model[exp] !== null && model[exp][metric] !== undefined && model[exp][metric] !== null) {
            sum += model[exp][metric];
            count++;
        }
    });

    return count > 0 ? sum / count : 0;
}

// Get score color class
function getScoreClass(score) {
    if (score >= 70) return 'score-high';
    if (score >= 50) return 'score-medium';
    return 'score-low';
}

// Get rank badge HTML
function getRankBadge(rank) {
    const classes = ['rank-1', 'rank-2', 'rank-3', 'rank-other'];
    const className = rank <= 3 ? classes[rank - 1] : classes[3];
    return `<span class="rank-badge ${className}">${rank}</span>`;
}

// Get model type badge
function getTypeBadge(type) {
    const typeMap = {
        'proprietary': { class: 'type-proprietary', label: 'Proprietary' },
        'opensource': { class: 'type-opensource', label: 'Open Source' },
        'baseline': { class: 'type-baseline', label: 'Baseline' }
    };
    const info = typeMap[type] || { class: '', label: type };
    return `<span class="model-type ${info.class}">${info.label}</span>`;
}

// Format score
function formatScore(score) {
    return score !== undefined && score !== null ? score.toFixed(2) : 'N/A';
}

// Get score cell HTML
function getScoreCell(model, exp, metric) {
    if (!model[exp] || model[exp] === null || model[exp][metric] === undefined) {
        return '<td class="score-na">--</td>';
    }
    const score = model[exp][metric];
    return `<td class="score ${getScoreClass(score)}">${formatScore(score)}</td>`;
}

// Filter and sort data
function getFilteredData() {
    let filtered = [...leaderboardData];

    // Calculate averages for each model
    filtered = filtered.map(model => ({
        ...model,
        average: calculateAverage(model, currentMetric)
    }));

    // Sort by current column
    filtered.sort((a, b) => {
        let aVal, bVal;

        if (currentSort.column === 'avg') {
            aVal = a.average;
            bVal = b.average;
        } else if (currentSort.column === 'model') {
            aVal = a.model.toLowerCase();
            bVal = b.model.toLowerCase();
            return currentSort.ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        } else if (currentSort.column === 'type') {
            aVal = a.type.toLowerCase();
            bVal = b.type.toLowerCase();
            return currentSort.ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        } else {
            // Experiment column
            const [exp] = currentSort.column.split('_');
            aVal = a[currentSort.column] ? a[currentSort.column][currentMetric] : 0;
            bVal = b[currentSort.column] ? b[currentSort.column][currentMetric] : 0;
        }

        return currentSort.ascending ? aVal - bVal : bVal - aVal;
    });

    return filtered;
}

// Render leaderboard
function renderLeaderboard() {
    const tbody = document.getElementById('leaderboardBody');
    const data = getFilteredData();

    // Apply search filter
    const searchTerm = document.getElementById('modelSearch').value.toLowerCase();
    const filteredData = searchTerm
        ? data.filter(model => model.model.toLowerCase().includes(searchTerm))
        : data;

    // Separate baseline models from ranked models
    const rankedModels = filteredData.filter(model => model.type !== 'baseline');
    const baselineModels = filteredData.filter(model => model.type === 'baseline');

    // Render ranked models
    const rankedHTML = rankedModels.map((model, index) => {
        const rank = index + 1;
        return `
            <tr>
                <td>${getRankBadge(rank)}</td>
                <td class="model-name">${model.model}</td>
                <td>${getTypeBadge(model.type)}</td>
                <td class="score ${getScoreClass(model.average)}">${formatScore(model.average)}</td>
                ${getScoreCell(model, 'exp1_text', currentMetric)}
                ${getScoreCell(model, 'exp1_audio', currentMetric)}
                ${getScoreCell(model, 'exp1_both', currentMetric)}
                ${getScoreCell(model, 'exp2_text', currentMetric)}
                ${getScoreCell(model, 'exp2_audio', currentMetric)}
                ${getScoreCell(model, 'exp2_both', currentMetric)}
                ${getScoreCell(model, 'exp3_text', currentMetric)}
                ${getScoreCell(model, 'exp3_audio', currentMetric)}
                ${getScoreCell(model, 'exp3_both', currentMetric)}
                ${getScoreCell(model, 'exp4_audio', currentMetric)}
            </tr>
        `;
    }).join('');

    // Render baseline models without rank
    const baselineHTML = baselineModels.map((model) => {
        return `
            <tr class="baseline-row">
                <td><span class="baseline-label">Baseline</span></td>
                <td class="model-name">${model.model}</td>
                <td>${getTypeBadge(model.type)}</td>
                <td class="score ${getScoreClass(model.average)}">${formatScore(model.average)}</td>
                ${getScoreCell(model, 'exp1_text', currentMetric)}
                ${getScoreCell(model, 'exp1_audio', currentMetric)}
                ${getScoreCell(model, 'exp1_both', currentMetric)}
                ${getScoreCell(model, 'exp2_text', currentMetric)}
                ${getScoreCell(model, 'exp2_audio', currentMetric)}
                ${getScoreCell(model, 'exp2_both', currentMetric)}
                ${getScoreCell(model, 'exp3_text', currentMetric)}
                ${getScoreCell(model, 'exp3_audio', currentMetric)}
                ${getScoreCell(model, 'exp3_both', currentMetric)}
                ${getScoreCell(model, 'exp4_audio', currentMetric)}
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rankedHTML + baselineHTML;

    // Update sort indicators
    document.querySelectorAll('.leaderboard-table th').forEach(th => {
        th.classList.remove('active');
        const icon = th.querySelector('i');
        if (icon) icon.remove();
    });

    const activeHeader = document.querySelector(`[data-sort="${currentSort.column}"]`);
    if (activeHeader) {
        activeHeader.classList.add('active');
        const icon = document.createElement('i');
        icon.className = currentSort.ascending ? 'fas fa-sort-up' : 'fas fa-sort-down';
        activeHeader.appendChild(document.createTextNode(' '));
        activeHeader.appendChild(icon);
    }
}

// Handle column header click
function handleSort(column) {
    if (currentSort.column === column) {
        currentSort.ascending = !currentSort.ascending;
    } else {
        currentSort.column = column;
        currentSort.ascending = false;
    }
    renderLeaderboard();
}

// Handle metric filter change
function handleMetricChange(e) {
    currentMetric = e.target.value;
    renderLeaderboard();
}

// Handle experiment filter change
function handleExperimentChange(e) {
    currentExperiment = e.target.value;

    // Show/hide columns based on experiment filter
    const table = document.getElementById('leaderboardTable');
    const headers = table.querySelectorAll('th');
    const rows = table.querySelectorAll('tbody tr');

    // Column indices for each experiment
    const columnMap = {
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'exp1': [0, 1, 2, 3, 4, 5, 6],
        'exp2': [0, 1, 2, 3, 7, 8, 9],
        'exp3': [0, 1, 2, 3, 10, 11, 12],
        'exp4': [0, 1, 2, 3, 13]
    };

    const visibleColumns = columnMap[currentExperiment] || columnMap['all'];

    // Update header visibility
    headers.forEach((header, index) => {
        header.style.display = visibleColumns.includes(index) ? '' : 'none';
    });

    // Update row cell visibility
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        cells.forEach((cell, index) => {
            cell.style.display = visibleColumns.includes(index) ? '' : 'none';
        });
    });
}

// Handle search
function handleSearch() {
    renderLeaderboard();
}

// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtex = document.querySelector('.citation-box code').textContent;
    navigator.clipboard.writeText(bibtex).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Render initial leaderboard
    renderLeaderboard();

    // Add event listeners
    document.querySelectorAll('.leaderboard-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort));
    });

    document.getElementById('metricFilter').addEventListener('change', handleMetricChange);
    document.getElementById('experimentFilter').addEventListener('change', handleExperimentChange);
    document.getElementById('modelSearch').addEventListener('input', handleSearch);
});

