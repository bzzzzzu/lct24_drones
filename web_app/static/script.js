$(document).ready(function() {
    document.getElementById('selectFolderBtn').addEventListener('click', function() {
        let input = document.getElementById('fileFolderInput');
        input.removeAttribute('multiple');
        input.setAttribute('webkitdirectory', 'webkitdirectory');
        input.click();
    });

    document.getElementById('selectFilesBtn').addEventListener('click', function() {
        let input = document.getElementById('fileFolderInput');
        input.removeAttribute('webkitdirectory');
        input.setAttribute('multiple', 'multiple');
        input.click();
    });

    document.getElementById('fileFolderInput').addEventListener('change', function(event) {
        let input = event.target;
        let fileList = input.files;
        let selectedFilesList = document.getElementById('selectedFiles');
        selectedFilesList.innerHTML = '';

        for (let i = 0; i < fileList.length; i++) {
            let li = document.createElement('li');
            li.textContent = fileList[i].name;
            selectedFilesList.appendChild(li);
        }

        // Enable the submit button if files are selected
        if (fileList.length > 0) {
            document.getElementById('SubmitFormBtn').removeAttribute('hidden');
            document.getElementById('ChoosedFiles').removeAttribute('hidden');
        } else {
            document.getElementById('SubmitFormBtn').setAttribute('hidden', 'hidden');
            document.getElementById('ChoosedFiles').setAttribute('hidden', 'hidden');
        }
    });

    function checkVideoStatus() {
        $.getJSON('/check_status', function(data) {
            if (data.status === 'ready') {
                document.getElementById("saveBtn").removeAttribute('hidden');
                if (data.type === 'video'){
                    var videoSection = $('#video-section');
                    videoSection.empty();
                    videoSection.append('<video id="my-video" class="video-js vjs-default-skin" controls preload="auto" width="1000" height="650">' +
                                        '<source src="/video/' + encodeURIComponent(data.result_file_path) + '" type="video/mp4">' +
                                        '</video>');

                    var player = videojs('my-video');

                    async function getTimecodes() {
                        try {
                            const response = await fetch('/timecodes/' + encodeURIComponent(data.result_file_path), {
                                method: 'GET',
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            });

                            if (!response.ok) {
                                throw new Error(`HTTP error! Status: ${response.status}`);
                            }

                            const timecodes = await response.json();
                            console.log('Received timecodes:', timecodes);
                            return timecodes;
                        } catch (error) {
                            console.error('Error fetching timecodes:', error);
                        }
                    }

                    var timecodes;

                    function createTimecode(timecode) {
                        var timecodeElem = document.createElement('div');
                        timecodeElem.classList.add('timecode');
                        timecodeElem.classList.add(timecode.color);

                        var startPercent = (timecode.start_time / player.duration()) * 100;
                        var endPercent = (timecode.end_time / player.duration()) * 100;
                        timecodeElem.style.left = startPercent + '%';
                        timecodeElem.style.width = (endPercent - startPercent) + '%';

                        timecodeElem.addEventListener('click', function(event) {
                            player.currentTime(timecode.start_time);
                        });

                        return timecodeElem;
                    }

                    function colorizeProgressBar() {
                        var progressBar = player.controlBar.progressControl.seekBar.el();

                        var existingTimecodes = progressBar.querySelectorAll('.timecode');
                        existingTimecodes.forEach(function(timecode) {
                            timecode.remove();
                        });

                        for (var i = 0; i < timecodes.length; i++) {
                            var timecode = timecodes[i];
                            console.log(timecode);
                            var timecodeElem = createTimecode(timecode);
                            timecodeElem.dataset.timeStart = timecode.start_time;
                            progressBar.appendChild(timecodeElem);
                        }
                    }

                    player.on('loadedmetadata', async function() {
                        timecodes = await getTimecodes();
                        colorizeProgressBar();
                    });
                }
                else if (data.type === 'image'){
                    var videoSection = $('#video-section');
                    videoSection.empty();
                    videoSection.append('<img src="/image/' + encodeURIComponent(data.result_file_path) + '" alt="Upload video">');
                }
            } else {
                document.getElementById("saveBtn").setAttribute('hidden', 'hidden');
                setTimeout(checkVideoStatus, 1000);
            }
        }).fail(function() {
            console.error('Failed to load video status');
        });
    }

    checkVideoStatus();
});
