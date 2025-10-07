document.getElementById('generate-summary').addEventListener('click', function() {
    const url = document.getElementById('youtube-url').value;
    if (url) {
        document.getElementById('spinner').style.display = 'inline-block';

        fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById('transcript').innerText = data.transcript;
                document.getElementById('summary').innerText = data.summary;

                // Update the AI Chat section
                document.getElementById('aichat').innerHTML = `
                    <div class="input-group">
                        <input type="text" id="ai-question" placeholder="Ask a question about the transcript">
                        <button id="ask-ai">Ask</button>
                    </div>
                    <div id="ai-response">
                        <!-- AI Chat response goes here -->
                    </div>
                `;

                // Add event listener for the Ask button in AI Chat
                document.getElementById('ask-ai').addEventListener('click', function() {
                    const question = document.getElementById('ai-question').value;
                    if (question) {
                        fetch('/ask', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ question: question })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert(data.error);
                            } else {
                                document.getElementById('ai-response').innerText = data.answer;
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('An error occurred while asking the question.');
                        });
                    } else {
                        alert('Please enter a question.');
                    }
                });

                // Show the video-summary section
                document.querySelector('.video-summary').classList.remove('hidden');

                // Optionally set the default tab
                document.querySelector('.tablink.active').click();
            }
            document.getElementById('spinner').style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching the summary.');
            document.getElementById('spinner').style.display = 'none';
        });
    } else {
        alert('Please enter a valid YouTube URL.');
    }
});

function openTab(evt, tabName) {
    const tablinks = document.getElementsByClassName('tablink');
    const tabcontents = document.getElementsByClassName('tabcontent');

    for (let i = 0; i < tabcontents.length; i++) {
        tabcontents[i].classList.remove('active');
    }

    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove('active');
    }

    document.getElementById(tabName).classList.add('active');
    evt.currentTarget.classList.add('active');
}
