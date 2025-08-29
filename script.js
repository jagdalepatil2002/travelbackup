document.addEventListener('DOMContentLoaded', () => {
    const searchButton = document.getElementById('search-button');
    const locationInput = document.getElementById('location-input');
    const resultsContainer = document.getElementById('results-container');
    const loader = document.getElementById('loader');
    
    const modal = document.getElementById('details-modal');
    const modalBody = document.getElementById('modal-body');
    const modalLoader = document.getElementById('modal-loader');
    const closeButton = document.querySelector('.close-button');
    const listenButton = document.getElementById('listen-button');

    const API_BASE_URL = 'https://backendtravel-pgzt.onrender.com';

    // --- Event Listeners ---

    searchButton.addEventListener('click', searchForPlaces);
    locationInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchForPlaces();
        }
    });

    closeButton.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    });

    listenButton.addEventListener('click', () => {
        // Placeholder for future text-to-speech functionality
        alert('Text-to-speech feature coming soon!');
    });

    // --- Functions ---

    async function searchForPlaces() {
        const location = locationInput.value.trim();
        if (!location) {
            alert('Please enter a location.');
            return;
        }

        resultsContainer.innerHTML = '';
        loader.style.display = 'block';

        try {
            const response = await fetch(`${API_BASE_URL}/search-places`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ location: location })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`Search Places call used ${data.token_count} tokens.`);
            displayPlaces(data.places);

        } catch (error) {
            console.error('Error fetching places:', error);
            resultsContainer.innerHTML = '<p class="error-message">Sorry, something went wrong. Please try again.</p>';
        } finally {
            loader.style.display = 'none';
        }
    }

    function displayPlaces(places) {
        if (!places || places.length === 0) {
            resultsContainer.innerHTML = '<p>No famous places found. Try a different search.</p>';
            return;
        }

        resultsContainer.innerHTML = ''; // Clear previous results
        places.forEach(place => {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            // Use a fallback image if image_url is null or undefined
            const imageUrl = place.image_url ? place.image_url : `https://picsum.photos/seed/${encodeURIComponent(place.name)}/400/300`;

            card.innerHTML = `
                <img src="${imageUrl}" alt="Image of ${place.name}">
                <div class="card-content">
                    <h3>${place.name}</h3>
                    <p>${place.description}</p>
                </div>
            `;
            card.addEventListener('click', () => showPlaceDetails(place.name));
            resultsContainer.appendChild(card);
        });
    }

    async function showPlaceDetails(placeName) {
        modal.style.display = 'flex';
        modalBody.innerHTML = '';
        modalLoader.style.display = 'block';

        try {
            const response = await fetch(`${API_BASE_URL}/place-details`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ place_name: placeName })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.token_count === 0) {
                console.log('Place Details loaded from cache (0 tokens used).');
            } else {
                console.log(`Place Details call used ${data.token_count} tokens.`);
            }

            // Simple markdown-like formatting for the output
            const formattedDescription = data.description
                .replace(/\*\*(.*?)\*\*/g, '<h2>$1</h2>') // Bold sections as subheadings
                .replace(/\n/g, '<br>'); // Newlines to line breaks

            modalBody.innerHTML = `<p>${formattedDescription}</p>`;

        } catch (error) {
            console.error('Error fetching place details:', error);
            modalBody.innerHTML = '<p class="error-message">Could not load details. Please try again later.</p>';
        } finally {
            modalLoader.style.display = 'none';
        }
    }
});
