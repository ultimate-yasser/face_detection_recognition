% Modular face detection using skin color + scoring system

function runFaceDetection()
    cam = initializeWebcam();
    videoPlayer = initializeVideoPlayer(cam);
    runLoop = true;

    % Set validation rule toggles
    isCheckOval = true;

    while runLoop
        frame = snapshot(cam);
        objects = detectSkinObjects(frame);

        for i = 1:length(objects)
            objects(i).points = 0;
            if isCheckOval && isOvalLikeShape(objects(i))
                objects(i).points = objects(i).points + 1;
            end
            % Add more rules here (e.g., eye detection, symmetry, etc.)
        end

        maxPts = max([objects.points]);
        bestObjects = objects([objects.points] == maxPts);

        for i = 1:length(bestObjects)
            bbox = bestObjects(i).BoundingBox;
            frame = insertShape(frame, 'Rectangle', bbox, 'LineWidth', 3);
        end

        step(videoPlayer, frame);
        runLoop = isOpen(videoPlayer);
    end

    clear cam;
    release(videoPlayer);
end

function cam = initializeWebcam()
    cam = webcam();
end

function videoPlayer = initializeVideoPlayer(cam)
    frame = snapshot(cam);
    frameSize = size(frame);
    videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)] + 30]);
end

function objects = detectSkinObjects(frame)
    imgYCbCr = rgb2ycbcr(frame);
    Cb = imgYCbCr(:,:,2);
    Cr = imgYCbCr(:,:,3);

    skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    skinMask = medfilt2(skinMask, [5 5]);
    skinMask = imfill(skinMask, 'holes');
    skinMask = bwareaopen(skinMask, 100);

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

function isValid = isOvalLikeShape(obj)
    % Placeholder rule for face-like shape
    isValid = true;
end
