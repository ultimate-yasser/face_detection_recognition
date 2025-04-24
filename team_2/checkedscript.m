function runFaceDetection()
    cam = initializeWebcam();
    videoPlayer = initializeVideoPlayer(cam);
    runLoop = true;

    % Set validation rule toggles
    iseye = true;

    while runLoop
        frame = snapshot(cam);
        objects = detectSkinObjects(frame);

        for i = 1:length(objects)
            objects(i).points = 0;

            % Crop the region of interest (ROI) from the frame
            bbox = objects(i).BoundingBox;
            roi = imcrop(frame, bbox);

            % Eye detection and scoring
            if iseye && iseyes(roi)  % Pass the cropped region to iseyes
                objects(i).points = objects(i).points + 1;
            end
        end

        maxPts = max([objects.points]);
        if maxPts > 0
            bestObjects = objects([objects.points] == maxPts);
            for i = 1:length(bestObjects)
                bbox = bestObjects(i).BoundingBox;
                frame = insertShape(frame, 'Rectangle', bbox, 'LineWidth', 3);
            end
        end

        step(videoPlayer, frame);
        runLoop = isOpen(videoPlayer);
    end

    clear cam;
    release(videoPlayer);
end

function cam = initializeWebcam()
    cam = webcam();  % Initialize the webcam
end

function videoPlayer = initializeVideoPlayer(cam)
    frame = snapshot(cam);  % Get a snapshot from the webcam
    frameSize = size(frame);  % Get the size of the frame
    videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)] + 30]);  % Initialize the video player
end

function objects = detectSkinObjects(frame)
    % Convert to YCbCr color space to detect skin tones
    imgYCbCr = rgb2ycbcr(frame);
    Cb = imgYCbCr(:,:,2);
    Cr = imgYCbCr(:,:,3);

    % Create skin mask
    skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    skinMask = medfilt2(skinMask, [5 5]);  % Median filter
    skinMask = imfill(skinMask, 'holes');  % Fill holes
    skinMask = bwareaopen(skinMask, 100);  % Remove small regions

    stats = regionprops(skinMask, 'BoundingBox', 'Area', 'Eccentricity');
    minArea = 500;
    objects = struct('BoundingBox', {}, 'points', {});
    for i = 1:length(stats)
        if stats(i).Area > minArea
            objects(end+1).BoundingBox = stats(i).BoundingBox;
            objects(end).points = 0;
        end
    end
end

function isValid = iseyes(roi)
    % Convert ROI to YCbCr
    imgYCbCr = rgb2ycbcr(roi);
    Y = imgYCbCr(:,:,1);
    Cb = imgYCbCr(:,:,2);
    Cr = imgYCbCr(:,:,3);

    % Detect dark regions from luminance (slightly relaxed)
    darkMask = Y < 85;

    % Skin detection using same YCbCr thresholds
    skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 150 & Cr <= 173);

    % Combine dark and skin regions
    eyeCandidateMask = darkMask & skinMask;

    % Morphological filtering
    eyeCandidateMask = bwareaopen(eyeCandidateMask, 12);
    eyeCandidateMask = imclose(eyeCandidateMask, strel('disk', 2));

    % Analyze connected components
    stats = regionprops(eyeCandidateMask, 'Centroid', 'BoundingBox', 'Eccentricity');
    filteredCenters = [];

    imgW = size(roi,2);
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox;
        aspectRatio = bbox(3) / bbox(4);  % width / height

        if aspectRatio > 0.4 && aspectRatio < 3.2 && stats(i).Eccentricity < 0.96
            cx = stats(i).Centroid(1);
            if cx > 0.10 * imgW && cx < 0.90 * imgW  % Looser side crop
                filteredCenters = [filteredCenters; stats(i).Centroid];
            end
        end
    end

    % Check for eye-like pairs
    eyePairs = [];
    for i = 1:size(filteredCenters, 1)
        for j = i+1:size(filteredCenters, 1)
            pt1 = filteredCenters(i,:);
            pt2 = filteredCenters(j,:);
            dist = norm(pt1 - pt2);
            if dist > 12 && dist < 130 && abs(pt1(2) - pt2(2)) < 35 && abs(pt1(1) - pt2(1)) > 10
                eyePairs = [eyePairs; pt1, pt2];
            end
        end
    end

    isValid = ~isempty(eyePairs);
end
