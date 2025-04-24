function runFaceDetection()
    % Initialize webcam with preferred resolution
    cam = webcam();
    
    % Create figure with reduced overhead
    fig = figure('Name', 'Face Detection', 'NumberTitle', 'off', ...
                 'DockControls', 'off', 'MenuBar', 'none');
    ax = axes('Parent', fig);
    hImg = imshow(zeros(480, 640, 3, 'uint8'), 'Parent', ax);
    
    % FPS control parameters
    targetFPS = 45;                    % Increased target FPS
    frameInterval = 1/targetFPS;
    lastFrameTime = tic;
    frameCount = 0;
    fps = targetFPS;
    
    % Preallocate buffers and convert to single precision
    prevTime = tic;
    [Y, Cb, Cr] = deal(single(zeros(480, 640)));  % Adjust resolution to match camera
    
    % Main processing loop
    while ishandle(fig)
        % Capture frame
        frame = snapshot(cam);
        
        % FAST face detection with vectorized operations
        faceBox = detectSingleFaceFast(frame, Y, Cb, Cr);
        
        % Optimized rectangle drawing
        if ~isempty(faceBox)
            frame = fastDrawRect(frame, faceBox);
        end
        
        % Update display
        set(hImg, 'CData', frame);
        drawnow('limitrate');
        
        % Dynamic FPS adjustment
        frameCount = frameCount + 1;
        if mod(frameCount, 15) == 0
            fps = 15/toc(prevTime);
            prevTime = tic;
            set(fig, 'Name', sprintf('Face Detection - %.1f FPS', fps));
        end
        
        % Precise timing control
        while toc(lastFrameTime) < frameInterval, end
        lastFrameTime = tic;
    end
    
    % Cleanup
    clear cam;
    close(fig);
end

function faceBox = detectSingleFaceFast(frame, Y, Cb, Cr)
    % Vectorized color conversion (single precision)
    R = single(frame(:,:,1));
    G = single(frame(:,:,2));
    B = single(frame(:,:,3));
    
    % In-place calculations to reduce memory allocation
    Y = 0.299*R + 0.587*G + 0.114*B;
    Cb = -0.1687*R - 0.3313*G + 0.5*B + 128;
    Cr  = 0.5*R - 0.4187*G - 0.0813*B + 128;
    
    % Vectorized skin mask
    skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    
    % Optimized median filtering
    skinMask = medfilt2(skinMask, [3 3]);
    
    % Fast connected components analysis
    [labeled, num] = bwlabel(skinMask, 8);
    if num == 0
        faceBox = [];
        return;
    end
    
    % Vectorized area calculation
    stats = regionprops(labeled, 'Area', 'BoundingBox');
    [maxArea, idx] = max([stats.Area]);
    
    if maxArea < 500
        faceBox = [];
    else
        faceBox = stats(idx).BoundingBox;
    end
end

function frame = fastDrawRect(frame, bbox)
    % Vectorized rectangle drawing using logical indexing
    x = round(bbox(1));
    y = round(bbox(2));
    w = round(bbox(3));
    h = round(bbox(4));
    
    % Calculate coordinates once
    [height, width, ~] = size(frame);
    border = 2;  % 3px border (0-2)
    
    % Horizontal lines
    y_top = y:y+border;
    y_bottom = y+h-border:y+h;
    valid_y_top = y_top >= 1 & y_top <= height;
    valid_y_bottom = y_bottom >= 1 & y_bottom <= height;
    
    for dy = y_top(valid_y_top)
        x_range = max(1,x):min(width,x+w);
        frame(dy, x_range, 1) = 0;   % Red channel
        frame(dy, x_range, 2) = 255; % Green channel
        frame(dy, x_range, 3) = 0;   % Blue channel
    end
    
    for dy = y_bottom(valid_y_bottom)
        x_range = max(1,x):min(width,x+w);
        frame(dy, x_range, 1) = 0;
        frame(dy, x_range, 2) = 255;
        frame(dy, x_range, 3) = 0;
    end
    
    % Vertical lines
    x_left = x:x+border;
    x_right = x+w-border:x+w;
    valid_x_left = x_left >= 1 & x_left <= width;
    valid_x_right = x_right >= 1 & x_right <= width;
    
    for dx = x_left(valid_x_left)
        y_range = max(1,y):min(height,y+h);
        frame(y_range, dx, 1) = 0;
        frame(y_range, dx, 2) = 255;
        frame(y_range, dx, 3) = 0;
    end
    
    for dx = x_right(valid_x_right)
        y_range = max(1,y):min(height,y+h);
        frame(y_range, dx, 1) = 0;
        frame(y_range, dx, 2) = 255;
        frame(y_range, dx, 3) = 0;
    end
end