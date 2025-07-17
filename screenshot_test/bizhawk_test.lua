-- Simple BizHawk Lua script to test frame capture
-- Load this in BizHawk: Tools -> Lua Console -> Script -> Open

local frame_count = 0
local output_dir = "C:\\Users\\richa\\Desktop\\Personal\\Uni\\ShenLong\\screenshot_test\\"

-- Function to save a single frame
function save_frame()
    frame_count = frame_count + 1
    
    -- Get current frame as bitmap data
    local start_time = os.clock()
    
    -- Method 1: Try client.screenshot() if available
    if client.screenshot then
        local success, result = pcall(function()
            return client.screenshot()
        end)
        
        if success then
            -- Save the screenshot
            local filename = output_dir .. "bizhawk_frame_" .. frame_count .. ".png"
            
            -- Try to save the screenshot
            local save_success, save_result = pcall(function()
                client.screenshot(filename)
            end)
            
            local end_time = os.clock()
            local elapsed_ms = (end_time - start_time) * 1000
            
            if save_success then
                console.log("Frame " .. frame_count .. " saved in " .. string.format("%.3f", elapsed_ms) .. "ms")
                console.log("File: " .. filename)
            else
                console.log("Failed to save frame: " .. tostring(save_result))
            end
        else
            console.log("client.screenshot() failed: " .. tostring(result))
        end
    else
        console.log("client.screenshot() not available")
    end
    
    -- Method 2: Try alternative frame access methods
    if emu and emu.getscreenshot then
        local success, result = pcall(function()
            return emu.getscreenshot()
        end)
        
        if success then
            console.log("emu.getscreenshot() available, data type: " .. type(result))
        else
            console.log("emu.getscreenshot() failed: " .. tostring(result))
        end
    end
end

-- Function to test frame capture timing
function test_capture_speed()
    console.log("=== BizHawk Frame Capture Test ===")
    console.log("Output directory: " .. output_dir)
    console.log("Testing frame capture capabilities...")
    
    -- Test what functions are available
    console.log("\nAvailable functions:")
    if client then
        console.log("- client table available")
        if client.screenshot then
            console.log("  - client.screenshot() available")
        else
            console.log("  - client.screenshot() NOT available")
        end
    else
        console.log("- client table NOT available")
    end
    
    if emu then
        console.log("- emu table available")
        if emu.getscreenshot then
            console.log("  - emu.getscreenshot() available")
        else
            console.log("  - emu.getscreenshot() NOT available")
        end
    else
        console.log("- emu table NOT available")
    end
    
    -- Try to capture one frame
    console.log("\nAttempting to capture frame...")
    save_frame()
    
    console.log("\nTest complete!")
    console.log("Check the screenshot_test folder for saved files.")
end

-- Main execution
test_capture_speed()

-- Register frame callback for continuous capture (commented out for now)
--[[
emu.registerafter(function()
    if frame_count < 10 then  -- Only capture first 10 frames
        save_frame()
    end
end)
--]]

console.log("\nTo capture frames continuously, uncomment the emu.registerafter section")
console.log("Press F5 to reload this script")