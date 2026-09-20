local mod = RegisterMod("FlyAI Combat Bridge", 1)

local game = Game()

local PREFIX = "FLYCOMBAT|"

local HIT_REWARD = 0.35
local MISS_REWARD = -0.05

-- ~10 updates/sec at ~30 game updates/sec.
local STATE_INTERVAL = 3

-- Use the REAL game collision radius.
-- Entity.Size is the collision radius and SizeMulti makes it elliptical.
-- Do not shrink it for the retina; discretization is handled in Python.
local SENSORY_HITBOX_SCALE = 1.00

local trackedTears = {}
local shotGroups = {}


-- ============================================================
-- OUTPUT
-- ============================================================

local function emit(text)
    Isaac.DebugString(PREFIX .. text)
end


-- ============================================================
-- HELPERS
-- ============================================================

local function currentRoomIndex()
    return game:GetLevel():GetCurrentRoomIndex()
end


local function actionFromVelocity(velocity)
    local x = velocity.X
    local y = velocity.Y

    if math.abs(x) > math.abs(y) then
        if x < 0 then
            return "shoot_left"
        else
            return "shoot_right"
        end
    else
        if y < 0 then
            return "shoot_up"
        else
            return "shoot_down"
        end
    end
end


-- ============================================================
-- NORMALIZED ENTITY POSITION + HITBOX
-- ============================================================

local function normalizedEntityHitbox(entity)
    local room = game:GetRoom()

    local topLeft = room:GetTopLeftPos()
    local bottomRight = room:GetBottomRightPos()

    local width = math.max(
        1.0,
        bottomRight.X - topLeft.X
    )

    local height = math.max(
        1.0,
        bottomRight.Y - topLeft.Y
    )

    local x = (
        entity.Position.X - topLeft.X
    ) / width

    local y = (
        entity.Position.Y - topLeft.Y
    ) / height

    x = math.max(
        0.0,
        math.min(
            1.0,
            x
        )
    )

    y = math.max(
        0.0,
        math.min(
            1.0,
            y
        )
    )

    local multiX = 1.0
    local multiY = 1.0

    if entity.SizeMulti ~= nil then
        multiX = math.abs(entity.SizeMulti.X)
        multiY = math.abs(entity.SizeMulti.Y)
    end

    local size = entity.Size or 0.0

    if size < 0.0 then
        size = 0.0
    end

    local radiusX = (
        size
        * multiX
        * SENSORY_HITBOX_SCALE
    ) / width

    local radiusY = (
        size
        * multiY
        * SENSORY_HITBOX_SCALE
    ) / height

    -- IMPORTANT: valid Lua multi-return.
    return x, y, radiusX, radiusY
end


-- ============================================================
-- EXACT GRID-SPACE HITBOX
-- ============================================================
--
-- This is the important accuracy path.
--
-- normalizedEntityHitbox() is useful as a backwards-compatible fallback,
-- but it can drift relative to the 2D room map because it normalizes
-- between room bounds.
--
-- Here we anchor the entity to Isaac's ACTUAL room grid using:
--     Room:GetClampedGridIndex()
--     Room:GetGridPosition()
--
-- gx / gy are floating RAW-grid coordinates, where a cell center is
-- col+0.5 / row+0.5.
--
-- grx / gry are the real physical hitbox radii measured in grid-cell units.
-- ============================================================

local function gridEntityHitbox(entity)
    local room = game:GetRoom()

    local gridWidth = math.max(
        1,
        room:GetGridWidth()
    )

    local gridIndex =
        room:GetClampedGridIndex(
            entity.Position
        )

    local row = math.floor(
        gridIndex / gridWidth
    )

    local col =
        gridIndex % gridWidth

    local center =
        room:GetGridPosition(
            gridIndex
        )

    -- Derive actual grid-cell spacing from Isaac itself.
    local p0 =
        room:GetGridPosition(0)

    local px =
        room:GetGridPosition(1)

    local py =
        room:GetGridPosition(
            gridWidth
        )

    local tileWidth = math.abs(
        px.X - p0.X
    )

    local tileHeight = math.abs(
        py.Y - p0.Y
    )

    if tileWidth < 0.001 then
        tileWidth = 40.0
    end

    if tileHeight < 0.001 then
        tileHeight = 40.0
    end

    local gx =
        col
        + 0.5
        + (
            entity.Position.X
            - center.X
        ) / tileWidth

    local gy =
        row
        + 0.5
        + (
            entity.Position.Y
            - center.Y
        ) / tileHeight

    local multiX = 1.0
    local multiY = 1.0

    if entity.SizeMulti ~= nil then
        multiX = math.abs(
            entity.SizeMulti.X
        )

        multiY = math.abs(
            entity.SizeMulti.Y
        )
    end

    local size =
        math.max(
            0.0,
            entity.Size or 0.0
        )

    local grx =
        (
            size
            * multiX
            * SENSORY_HITBOX_SCALE
        ) / tileWidth

    local gry =
        (
            size
            * multiY
            * SENSORY_HITBOX_SCALE
        ) / tileHeight

    return gx, gy, grx, gry
end


-- ============================================================
-- EXACT SCREEN-SPACE HITBOX
-- ============================================================
--
-- Used only for dashboard auditing. These values let Python draw the
-- true collision ellipse directly over the captured Isaac frame.
-- ============================================================

local function screenEntityHitbox(entity)
    local screenWidth =
        math.max(
            1.0,
            Isaac.GetScreenWidth()
        )

    local screenHeight =
        math.max(
            1.0,
            Isaac.GetScreenHeight()
        )

    local screenPos =
        Isaac.WorldToScreen(
            entity.Position
        )

    local multiX = 1.0
    local multiY = 1.0

    if entity.SizeMulti ~= nil then
        multiX = math.abs(
            entity.SizeMulti.X
        )

        multiY = math.abs(
            entity.SizeMulti.Y
        )
    end

    local size =
        math.max(
            0.0,
            entity.Size or 0.0
        )

    local dx =
        Isaac.WorldToScreenDistance(
            Vector(
                size
                * multiX
                * SENSORY_HITBOX_SCALE,
                0.0
            )
        )

    local dy =
        Isaac.WorldToScreenDistance(
            Vector(
                0.0,
                size
                * multiY
                * SENSORY_HITBOX_SCALE
            )
        )

    local sx =
        screenPos.X
        / screenWidth

    local sy =
        screenPos.Y
        / screenHeight

    local srx =
        math.abs(
            dx.X
        ) / screenWidth

    local sry =
        math.abs(
            dy.Y
        ) / screenHeight

    return sx, sy, srx, sry
end


-- ============================================================
-- PLAYER + ENEMY VISION
-- ============================================================

local function getEnemyVision()
    local player =
        Isaac.GetPlayer(0)

    local playerX = 0.5
    local playerY = 0.5
    local playerRX = 0.015
    local playerRY = 0.025

    local playerGX = 0.0
    local playerGY = 0.0
    local playerGRX = 0.0
    local playerGRY = 0.0

    local playerSX = 0.5
    local playerSY = 0.5
    local playerSRX = 0.0
    local playerSRY = 0.0

    if player ~= nil then
        playerX,
        playerY,
        playerRX,
        playerRY =
            normalizedEntityHitbox(
                player
            )

        playerGX,
        playerGY,
        playerGRX,
        playerGRY =
            gridEntityHitbox(
                player
            )

        playerSX,
        playerSY,
        playerSRX,
        playerSRY =
            screenEntityHitbox(
                player
            )
    end

    local enemyData = {}
    local visibleEnemies = 0

    local entities =
        Isaac.GetRoomEntities()

    for _, entity
        in ipairs(
            entities
        )
    do
        local npc =
            entity:ToNPC()

        if npc ~= nil
            and npc:IsEnemy()
            and npc:IsActiveEnemy(false)
            and not npc:IsDead()
        then
            local x,
                  y,
                  rx,
                  ry =
                normalizedEntityHitbox(
                    npc
                )

            local gx,
                  gy,
                  grx,
                  gry =
                gridEntityHitbox(
                    npc
                )

            local sx,
                  sy,
                  srx,
                  sry =
                screenEntityHitbox(
                    npc
                )

            local vulnerable = 0

            if npc:IsVulnerableEnemy() then
                vulnerable = 1
            end

            local boss = 0

            if npc:IsBoss() then
                boss = 1
            end

            visibleEnemies =
                visibleEnemies
                + 1

            table.insert(
                enemyData,
                string.format(
                    "%.5f,%.5f,%.6f,%.6f,%d,%d,"
                    .. "%.5f,%.5f,%.6f,%.6f,"
                    .. "%.5f,%.5f,%.6f,%.6f",
                    x,
                    y,
                    rx,
                    ry,
                    vulnerable,
                    boss,
                    gx,
                    gy,
                    grx,
                    gry,
                    sx,
                    sy,
                    srx,
                    sry
                )
            )
        end
    end

    local enemyString = "-"

    if #enemyData > 0 then
        enemyString =
            table.concat(
                enemyData,
                ";"
            )
    end

    return
        playerX,
        playerY,
        playerRX,
        playerRY,
        playerGX,
        playerGY,
        playerGRX,
        playerGRY,
        playerSX,
        playerSY,
        playerSRX,
        playerSRY,
        visibleEnemies,
        enemyString
end


-- ============================================================
-- HOSTILE PROJECTILES
-- ============================================================

local function projectileState()
    local player = Isaac.GetPlayer(0)

    if player == nil then
        return 0, -1.0, 0.0, 0.0
    end

    local entities = Isaac.FindByType(
        EntityType.ENTITY_PROJECTILE,
        -1,
        -1,
        false,
        false
    )

    local count = 0
    local nearestDistance = nil
    local nearestDX = 0.0
    local nearestDY = 0.0

    for _, entity in ipairs(entities) do
        local projectile = entity:ToProjectile()

        if projectile ~= nil then
            local friendly = false
            local spawner = projectile.SpawnerEntity

            if spawner ~= nil
                and spawner:ToPlayer() ~= nil
            then
                friendly = true
            end

            if not friendly then
                count = count + 1

                local delta = (
                    projectile.Position
                    - player.Position
                )

                local distance = delta:Length()

                if nearestDistance == nil
                    or distance < nearestDistance
                then
                    nearestDistance = distance
                    nearestDX = delta.X
                    nearestDY = delta.Y
                end
            end
        end
    end

    if nearestDistance == nil then
        nearestDistance = -1.0
    end

    -- IMPORTANT: valid Lua multi-return.
    return count, nearestDistance, nearestDX, nearestDY
end


-- ============================================================
-- PERIODIC STATE
-- ============================================================

local function emitCombatState()
    local room =
        game:GetRoom()

    local aliveEnemies =
        room:GetAliveEnemiesCount()

    local projectileCount,
          projectileDistance,
          projectileDX,
          projectileDY =
        projectileState()

    local playerX,
          playerY,
          playerRX,
          playerRY,
          playerGX,
          playerGY,
          playerGRX,
          playerGRY,
          playerSX,
          playerSY,
          playerSRX,
          playerSRY,
          visibleEnemies,
          enemyData =
        getEnemyVision()

    emit(
        "STATE|"
        .. tostring(
            aliveEnemies
        )
        .. "|"
        .. tostring(
            projectileCount
        )
        .. "|"
        .. string.format(
            "%.3f",
            projectileDistance
        )
        .. "|"
        .. string.format(
            "%.3f",
            projectileDX
        )
        .. "|"
        .. string.format(
            "%.3f",
            projectileDY
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerX
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerY
        )
        .. "|"
        .. tostring(
            visibleEnemies
        )
        .. "|"
        .. enemyData
        .. "|"
        .. string.format(
            "%.6f",
            playerRX
        )
        .. "|"
        .. string.format(
            "%.6f",
            playerRY
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerGX
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerGY
        )
        .. "|"
        .. string.format(
            "%.6f",
            playerGRX
        )
        .. "|"
        .. string.format(
            "%.6f",
            playerGRY
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerSX
        )
        .. "|"
        .. string.format(
            "%.5f",
            playerSY
        )
        .. "|"
        .. string.format(
            "%.6f",
            playerSRX
        )
        .. "|"
        .. string.format(
            "%.6f",
            playerSRY
        )
    )
end


-- ============================================================
-- TEAR TRACKING
-- ============================================================

local function onFireTear(_, tear)
    if tear == nil then
        return
    end

    local action =
        actionFromVelocity(
            tear.Velocity
        )

    local frame =
        game:GetFrameCount()

    local roomIndex =
        currentRoomIndex()

    local enemyCount =
        game:GetRoom():GetAliveEnemiesCount()

    local groupKey =
        tostring(frame)
        .. ":"
        .. action

    if shotGroups[groupKey] == nil then
        shotGroups[groupKey] = {
            active = 0,
            hit = false,
            action = action,
            room = roomIndex,
            hadEnemies = enemyCount > 0,
        }
    end

    local group =
        shotGroups[groupKey]

    group.active =
        group.active + 1

    trackedTears[
        GetPtrHash(tear)
    ] = groupKey
end


local function onTearCollision(
    _,
    tear,
    collider,
    low
)
    if tear == nil
        or collider == nil
    then
        return nil
    end

    local hash =
        GetPtrHash(tear)

    local groupKey =
        trackedTears[hash]

    if groupKey == nil then
        return nil
    end

    local group =
        shotGroups[groupKey]

    if group == nil then
        return nil
    end

    local npc =
        collider:ToNPC()

    if npc ~= nil
        and npc:IsEnemy()
        and npc:IsVulnerableEnemy()
        and not npc:IsDead()
    then
        if not group.hit then
            group.hit = true

            emit(
                "SHOT_HIT|"
                .. tostring(HIT_REWARD)
                .. "|"
                .. group.action
                .. "|enemy="
                .. tostring(npc.Type)
                .. ":"
                .. tostring(npc.Variant)
            )
        end
    end

    return nil
end


local function onTearRemoved(_, entity)
    if entity == nil then
        return
    end

    local hash =
        GetPtrHash(entity)

    local groupKey =
        trackedTears[hash]

    if groupKey == nil then
        return
    end

    trackedTears[hash] = nil

    local group =
        shotGroups[groupKey]

    if group == nil then
        return
    end

    group.active =
        group.active - 1

    if group.active > 0 then
        return
    end

    if not group.hit
        and group.hadEnemies
        and currentRoomIndex() == group.room
    then
        emit(
            "SHOT_MISS|"
            .. tostring(MISS_REWARD)
            .. "|"
            .. group.action
            .. "|"
        )
    end

    shotGroups[groupKey] = nil
end


-- ============================================================
-- ROOM / UPDATE
-- ============================================================

local function onNewRoom(_)
    trackedTears = {}
    shotGroups = {}

    emitCombatState()
end


local function onUpdate(_)
    if (
        game:GetFrameCount()
        % STATE_INTERVAL
    ) == 0 then
        emitCombatState()
    end
end


-- ============================================================
-- CALLBACKS
-- ============================================================

mod:AddCallback(
    ModCallbacks.MC_POST_FIRE_TEAR,
    onFireTear
)

mod:AddCallback(
    ModCallbacks.MC_PRE_TEAR_COLLISION,
    onTearCollision
)

mod:AddCallback(
    ModCallbacks.MC_POST_ENTITY_REMOVE,
    onTearRemoved,
    EntityType.ENTITY_TEAR
)

mod:AddCallback(
    ModCallbacks.MC_POST_NEW_ROOM,
    onNewRoom
)

mod:AddCallback(
    ModCallbacks.MC_POST_UPDATE,
    onUpdate
)

emit("BRIDGE_LOADED|4")
