# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球场地模块

包含:
- 物理场景设置
- 灯光环境
- 地面、球网、边线创建
- 材质管理
"""

from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
)


# =====================================================================
# 默认场地参数
# =====================================================================

DEFAULT_COURT_LENGTH = 18.0      # 全场长度 (m)
DEFAULT_COURT_WIDTH = 9.0        # 场地宽度 (m)
DEFAULT_NET_HEIGHT = 2.43        # 男子球网高度 (m)
DEFAULT_NET_THICKNESS = 0.05

# 物理材质参数
DEFAULT_FLOOR_STATIC_FRICTION = 0.60
DEFAULT_FLOOR_DYNAMIC_FRICTION = 0.50
DEFAULT_FLOOR_COR = 0.75


# =====================================================================
# 排球场地类
# =====================================================================

class VolleyballCourt:
    """
    排球场地管理类
    
    创建:
    - 物理场景（GPU dynamics）
    - 灯光环境
    - 地面与碰撞
    - 球网（可选物理碰撞）
    - 场地边线
    """
    
    def __init__(self, stage, 
                 court_length: float = DEFAULT_COURT_LENGTH,
                 court_width: float = DEFAULT_COURT_WIDTH,
                 net_height: float = DEFAULT_NET_HEIGHT,
                 enable_net_collision: bool = False):
        """
        初始化场地
        
        Args:
            stage: USD stage
            court_length: 场地长度 (X方向)
            court_width: 场地宽度 (Y方向)
            net_height: 球网高度
            enable_net_collision: 是否启用球网物理碰撞
        """
        self.stage = stage
        self.court_length = court_length
        self.court_width = court_width
        self.net_height = net_height
        self.enable_net_collision = enable_net_collision
        
        self._materials = {}
    
    def setup_all(self):
        """一键设置完整场地"""
        print("[VolleyballCourt] Setting up physics scene...")
        self.setup_physics_scene()
        print("[VolleyballCourt] Setting up lighting...")
        self.setup_environment()
        print("[VolleyballCourt] Creating ground plane...")
        self.create_ground()
        print(f"[VolleyballCourt] Creating court surface ({self.court_length}m x {self.court_width}m)...")
        self.create_court_surface()
        print("[VolleyballCourt] Creating court lines...")
        self.create_court_lines()
        print(f"[VolleyballCourt] Creating net (height={self.net_height}m, collision={self.enable_net_collision})...")
        self.create_net()
        print("[VolleyballCourt] Full court setup complete!")
    
    # ===== 物理场景 =====
    
    def setup_physics_scene(self):
        """设置物理场景（GPU dynamics）"""
        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/physicsScene"))
        physx_scene = PhysxSchema.PhysxSceneAPI.Get(self.stage, "/physicsScene")
        physx_scene.CreateEnableCCDAttr(True)
        physx_scene.CreateEnableStabilizationAttr(True)
        physx_scene.CreateEnableGPUDynamicsAttr(True)
        physx_scene.CreateSolverTypeAttr("TGS")
    
    def setup_environment(self):
        """设置灯光环境"""
        # 平行光（主光源）
        distant_light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/DistantLight"))
        distant_light.CreateIntensityAttr(800)
        distant_light.CreateAngleAttr(1.0)
        xform = UsdGeom.Xformable(distant_light)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
        
        # 环境光
        dome_light = UsdLux.DomeLight.Define(self.stage, Sdf.Path("/DomeLight"))
        dome_light.CreateIntensityAttr(200)
    
    # ===== 材质管理 =====
    
    def _create_visual_material(self, name: str, color: tuple, 
                                 roughness: float = 0.5, metallic: float = 0.0):
        """创建视觉材质"""
        if name in self._materials:
            return self._materials[name]
        
        path = f"/Materials/{name}"
        material = UsdShade.Material.Define(self.stage, Sdf.Path(path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{path}/Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        self._materials[name] = material
        return material
    
    def _create_physics_material(self, name: str, static_f: float, 
                                  dynamic_f: float, restitution: float):
        """创建物理材质"""
        if name in self._materials:
            return self._materials[name]
        
        path = f"/Materials/{name}"
        material = UsdShade.Material.Define(self.stage, Sdf.Path(path))
        physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_mat.CreateStaticFrictionAttr(static_f)
        physics_mat.CreateDynamicFrictionAttr(dynamic_f)
        physics_mat.CreateRestitutionAttr(restitution)
        
        self._materials[name] = material
        return material
    
    # ===== 地面 =====
    
    def create_ground(self):
        """创建物理地面"""
        PhysicsSchemaTools.addGroundPlane(
            self.stage, "/groundPlane", "Z", 1500, 
            Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.15, 0.15, 0.15)
        )
        
        # 绑定物理材质
        ground_phys_mat = self._create_physics_material(
            "GroundPhysMat",
            DEFAULT_FLOOR_STATIC_FRICTION,
            DEFAULT_FLOOR_DYNAMIC_FRICTION,
            DEFAULT_FLOOR_COR
        )
        
        ground_geom = self.stage.GetPrimAtPath("/groundPlane/geom")
        if ground_geom.IsValid():
            phys_binding = UsdShade.MaterialBindingAPI.Apply(ground_geom)
            phys_binding.Bind(ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    
    def create_court_surface(self):
        """创建球场地面（视觉+碰撞）"""
        court = UsdGeom.Cube.Define(self.stage, Sdf.Path("/World/Court"))
        court.CreateSizeAttr(1.0)
        
        xf = UsdGeom.Xformable(court)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.005))
        xf.AddScaleOp().Set(Gf.Vec3f(self.court_length, self.court_width, 0.01))
        
        # 视觉材质（木地板色）
        court_mat = self._create_visual_material("CourtMat", (0.82, 0.68, 0.46), roughness=0.7)
        UsdShade.MaterialBindingAPI(court).Bind(court_mat)
        
        # 碰撞
        UsdPhysics.CollisionAPI.Apply(court.GetPrim())
        
        # 物理材质
        ground_phys_mat = self._materials.get("GroundPhysMat")
        if ground_phys_mat:
            phys_binding = UsdShade.MaterialBindingAPI.Apply(court.GetPrim())
            phys_binding.Bind(ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    
    def create_court_lines(self):
        """创建球场边线"""
        line_mat = self._create_visual_material("LineMat", (1.0, 1.0, 1.0), roughness=0.9)
        lw = 0.05
        
        lines = [
            ("LineN", (0, self.court_width/2, 0.015), (self.court_length, lw, 0.005)),
            ("LineS", (0, -self.court_width/2, 0.015), (self.court_length, lw, 0.005)),
            ("LineE", (self.court_length/2, 0, 0.015), (lw, self.court_width, 0.005)),
            ("LineW", (-self.court_length/2, 0, 0.015), (lw, self.court_width, 0.005)),
            ("LineMid", (0, 0, 0.015), (lw, self.court_width, 0.005)),
        ]
        
        for name, pos, scale in lines:
            line = UsdGeom.Cube.Define(self.stage, Sdf.Path(f"/World/{name}"))
            line.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(line)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI(line).Bind(line_mat)
    
    # ===== 球网 =====
    
    def create_net(self):
        """创建球网"""
        # 球网主体
        net = UsdGeom.Cube.Define(self.stage, Sdf.Path("/World/Net/Mesh"))
        net.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(net)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, self.net_height / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(DEFAULT_NET_THICKNESS, self.court_width + 1.0, self.net_height))
        
        net_mat = self._create_visual_material("NetMat", (0.15, 0.15, 0.15), roughness=0.9)
        UsdShade.MaterialBindingAPI(net).Bind(net_mat)
        
        if self.enable_net_collision:
            UsdPhysics.CollisionAPI.Apply(net.GetPrim())
        
        # 顶带
        top = UsdGeom.Cube.Define(self.stage, Sdf.Path("/World/Net/TopBand"))
        top.CreateSizeAttr(1.0)
        xf2 = UsdGeom.Xformable(top)
        xf2.AddTranslateOp().Set(Gf.Vec3d(0, 0, self.net_height + 0.035))
        xf2.AddScaleOp().Set(Gf.Vec3f(DEFAULT_NET_THICKNESS + 0.02, self.court_width + 1.0, 0.07))
        
        top_mat = self._create_visual_material("TopBandMat", (0.95, 0.95, 0.95), roughness=0.8)
        UsdShade.MaterialBindingAPI(top).Bind(top_mat)
        
        # 柱子
        pole_mat = self._create_visual_material("PoleMat", (0.6, 0.6, 0.65), roughness=0.3, metallic=0.8)
        pole_h = self.net_height + 0.3
        
        for i, y in enumerate([-(self.court_width/2 + 0.5), (self.court_width/2 + 0.5)]):
            pole = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(f"/World/Net/Pole{i}"))
            pole.CreateRadiusAttr(0.04)
            pole.CreateHeightAttr(pole_h)
            pole.CreateAxisAttr("Z")
            xf = UsdGeom.Xformable(pole)
            xf.AddTranslateOp().Set(Gf.Vec3d(0, y, pole_h/2))
            UsdShade.MaterialBindingAPI(pole).Bind(pole_mat)
    
    # ===== 碰网检测 =====
    
    def check_net_collision(self, ball_pos, ball_radius: float = 0.105) -> bool:
        """几何检测球是否碰网"""
        x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        net_half_t = 0.15  # 碰网检测半厚
        
        return (abs(x) < net_half_t + ball_radius
                and abs(y) < (self.court_width / 2 + 1.0)
                and z < self.net_height + ball_radius)
    
    def check_out_of_bounds(self, ball_pos, margin: float = 1.2) -> bool:
        """检测球是否出界"""
        x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        
        oob_x = self.court_length * margin
        oob_y = self.court_width * 1.5
        oob_z_high = 15.0
        
        return abs(x) > oob_x or abs(y) > oob_y or z > oob_z_high
