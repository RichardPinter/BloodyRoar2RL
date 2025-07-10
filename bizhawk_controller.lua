-- Optimized controller with configurable polling rate
console.log("=== BizHawk Controller (Optimized) ===")

local ACTIONS_FILE = "C:\\Users\\richa\\Desktop\\Personal\\Uni\\ShenLong\\actions.txt"
local frame_counter = 0

-- CONFIGURE THIS: Lower = faster response but more CPU
local check_interval = 5  -- Check every 5 frames (~83ms at 60fps)
-- Options: 3 (fast), 5 (balanced), 10 (default), 30 (slow)

-- Performance stats
local actions_processed = 0
local start_time = os.clock()

function execute_action(action_text)
    -- Handle actions with timestamps (for testing)
    local action, timestamp = action_text:match("([^|]+)|?(.*)")
    action = action:lower():gsub("%s+", "")
    
    -- Clear previous inputs
    joypad.set({})
    
    if action == "kick" then
        console.log("KICK!")
        joypad.set({["P1 X"] = true})          -- X = Kick
        return true
    elseif action == "punch" then
        console.log("PUNCH!")
        joypad.set({["P1 ⬜"] = true})         -- Square = Punch
        return true
    elseif action == "transform" then
        console.log("TRANSFORM!")
        joypad.set({["P1 ○"] = true})          -- Circle = Transform
        return true
    elseif action == "special" then
        console.log("SPECIAL!")
        joypad.set({["P1 △"] = true})          -- Triangle = Special
        return true
    elseif action == "throw" then
        console.log("THROW!")
        joypad.set({["P1 R2"] = true})         -- R2 = Throw
        return true
    elseif action == "block" or action == "r1" then
        console.log("BLOCK!")
        joypad.set({["P1 R1"] = true})
        return true
    elseif action == "beast" or action == "l1+l2" then
        console.log("BEAST TRANSFORM!")
        joypad.set({["P1 L1"] = true, ["P1 L2"] = true})
        return true
    elseif action == "jump" or action == "up" then
        console.log("JUMP!")
        joypad.set({["P1 D-Pad Up"] = true})
        return true
    elseif action == "squat" or action == "down" then
        console.log("SQUAT!")
        joypad.set({["P1 D-Pad Down"] = true})
        return true
    elseif action == "left" then
        console.log("LEFT!")
        joypad.set({["P1 D-Pad Left"] = true})
        return true
    elseif action == "right" then
        console.log("RIGHT!")
        joypad.set({["P1 D-Pad Right"] = true})
        return true
    elseif action == "l1" then
        console.log("L1!")
        joypad.set({["P1 L1"] = true})
        return true
    elseif action == "l2" then
        console.log("L2!")
        joypad.set({["P1 L2"] = true})
        return true
    elseif action == "r2" then
        console.log("R2!")
        joypad.set({["P1 R2"] = true})
        return true
    elseif action == "start" then
        console.log("START!")
        joypad.set({["P1 Start"] = true})
        return true
    elseif action == "select" then
        console.log("SELECT!")
        joypad.set({["P1 Select"] = true})
        return true
    elseif action == "stop" then
        joypad.set({})
        return true
    else
        console.log("Unknown action: " .. action)
        return false
    end
end

console.log("Settings: Checking every " .. check_interval .. " frames")
console.log("Watching: " .. ACTIONS_FILE)
console.log("Ready for Python commands...")

while true do
    frame_counter = frame_counter + 1
    
    if frame_counter % check_interval == 0 then
        local file = io.open(ACTIONS_FILE, "r")
        if file then
            local content = file:read("*all")
            file:close()
            
            if content and content ~= "" then
                if execute_action(content) then
                    actions_processed = actions_processed + 1
                    
                    -- Clear file immediately
                    local clear_file = io.open(ACTIONS_FILE, "w")
                    if clear_file then
                        clear_file:write("")
                        clear_file:close()
                    end
                    
                    -- Show performance stats every 100 actions
                    if actions_processed % 100 == 0 then
                        local elapsed = os.clock() - start_time
                        local rate = actions_processed / elapsed
                        console.log(string.format("Performance: %.1f actions/sec", rate))
                    end
                end
            end
        end
    end
    
    emu.frameadvance()
end