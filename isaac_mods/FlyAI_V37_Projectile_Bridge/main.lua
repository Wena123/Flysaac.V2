local mod = RegisterMod("FlyAI V3.7 Projectile Bridge", 1)
local game = Game()
local PREFIX = "FLYCOMBATV37|"
local STATE_INTERVAL = 1       -- every game update (~30 Hz)
local MAX_PROJECTILES = 64
local GAME_UPDATES_PER_SECOND = 30.0

local function emit(text)
    Isaac.DebugString(PREFIX .. text)
end

local function roomMetrics()
    local room = game:GetRoom()
    local tl = room:GetTopLeftPos()
    local br = room:GetBottomRightPos()
    return tl, math.max(1.0, br.X-tl.X), math.max(1.0, br.Y-tl.Y)
end

local function normalizedHitbox(entity, tl, width, height)
    local x = math.max(0.0, math.min(1.0, (entity.Position.X-tl.X)/width))
    local y = math.max(0.0, math.min(1.0, (entity.Position.Y-tl.Y)/height))
    local mx, my = 1.0, 1.0
    if entity.SizeMulti ~= nil then mx=math.abs(entity.SizeMulti.X); my=math.abs(entity.SizeMulti.Y) end
    local size = math.max(0.0, entity.Size or 0.0)
    return x, y, size*mx/width, size*my/height
end

local function isFriendly(projectile)
    local spawner = projectile.SpawnerEntity
    return spawner ~= nil and spawner:ToPlayer() ~= nil
end

local function emitState()
    local player = Isaac.GetPlayer(0)
    if player == nil then return end
    local tl, width, height = roomMetrics()
    local px, py, prx, pry = normalizedHitbox(player, tl, width, height)
    local entities = Isaac.FindByType(EntityType.ENTITY_PROJECTILE, -1, -1, false, false)
    local rows = {}
    for _, entity in ipairs(entities) do
        local p = entity:ToProjectile()
        if p ~= nil and not isFriendly(p) then
            local x,y,rx,ry = normalizedHitbox(p, tl, width, height)
            local dx = p.Position.X-player.Position.X
            local dy = p.Position.Y-player.Position.Y
            local d2 = dx*dx+dy*dy
            local vx = p.Velocity.X/width*GAME_UPDATES_PER_SECOND
            local vy = p.Velocity.Y/height*GAME_UPDATES_PER_SECOND
            table.insert(rows, {d2=d2, text=string.format("%.5f,%.5f,%.6f,%.6f,%.6f,%.6f",x,y,vx,vy,rx,ry)})
        end
    end
    table.sort(rows, function(a,b) return a.d2 < b.d2 end)
    local out = {}
    local limit = math.min(#rows, MAX_PROJECTILES)
    for i=1,limit do out[i]=rows[i].text end
    local data = limit > 0 and table.concat(out, ";") or "-"
    local roomIndex = game:GetLevel():GetCurrentRoomIndex()
    emit(string.format("STATE|%d|%.5f|%.5f|%.6f|%.6f|%d|%s", roomIndex, px, py, prx, pry, limit, data))
end

local function onUpdate(_)
    if game:GetFrameCount() % STATE_INTERVAL == 0 then emitState() end
end
local function onNewRoom(_) emitState() end
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, onUpdate)
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, onNewRoom)
