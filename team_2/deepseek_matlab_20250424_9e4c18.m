function runFaceDetection()
    % Initialize webcam
    cam = webcam();
    
    % Create figure for display
    fig = figure('Name', 'Face Detection');
    set(fig, 'Position', [100 100 800 600]);
    ax = axes('Parent', fig);
    
    % Main loop
    while ishandle(fig)
        % Capture frame
        frame = snapshot(cam);
        
        % Detect face (returns empty if no face found)
        faceBox = detectSingleFace(frame);
        
        % Draw rectangle if face found
        if ~isempty(faceBox)
            frame = drawRectangle(frame, faceBox);
        end
        
        % Display result
        imshow(frame, 'Parent', ax);
        drawnow;
    end
    
    % Cleanup
    clear cam;
    close(fig);
end

function faceBox = detectSingleFace(frame)
    % Convert RGB to YCbCr manually
    R = double(frame(:,:,1));
    G = double(frame(:,:,2));
    B = double(frame(:,:,3));
    
    Y = 0.299*R + 0.587*G + 0.114*B;
    Cb = -0.1687*R - 0.3313*G + 0.5*B + 128;
    Cr = 0.5*R - 0.4187*G - 0.0813*B + 128;
    
    % Create skin color mask
    skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    
    % Simple noise reduction (3x3 median filter)
    skinMask = medfilt2_simple(skinMask, 3);
    
    % Find largest connected component
    [labeled, num] = bwlabel(skinMask, 8);
    if num == 0
        faceBox = [];
        return;
    end
    
    % Calculate area of each component
    areas = zeros(1, num);
    for i = 1:num
        areas(i) = sum(labeled(:) == i);
    end
    
    % Get largest component
    [maxArea, idx] = max(areas);
    if maxArea < 500  % Minimum face area threshold
        faceBox = [];
        return;
    end
    
    % Calculate bounding box
    [rows, cols] = find(labeled == idx);
    x1 = min(cols);
    x2 = max(cols);
    y1 = min(rows);
    y2 = max(rows);
    
    faceBox = [x1, y1, x2-x1, y2-y1];
end

function filtered = medfilt2_simple(img, windowSize)
    % Simple 3x3 median filter implementation
    [rows, cols] = size(img);
    filtered = false(rows, cols);
    pad = floor(windowSize/2);
    
    for i = 1+pad:rows-pad
        for j = 1+pad:cols-pad
            neighborhood = img(i-pad:i+pad, j-pad:j+pad);
            filtered(i,j) = median(neighborhood(:));
        end
    end
end

function frame = drawRectangle(frame, bbox)
    % Draw rectangle manually without insertShape
    x = round(bbox(1));
    y = round(bbox(2));
    w = round(bbox(3));
    h = round(bbox(4));
    
    % Draw green rectangle (3px width)
    color = [0 255 0]; % Green
    
    % Top and bottom lines
    for i = max(1,x):min(size(frame,2),x+w)
        for t = 0:2
            if y+t <= size(frame,1)
                frame(y+t,i,:) = color;
            end
            if y+h-t <= size(frame,1)
                frame(y+h-t,i,:) = color;
            end
        end
    end
    
    % Left and right lines
    for j = max(1,y):min(size(frame,1),y+h)
        for t = 0:2
            if x+t <= size(frame,2)
                frame(j,x+t,:) = color;
            end
            if x+w-t <= size(frame,2)
                frame(j,x+w-t,:) = color;
            end
        end
    end
end