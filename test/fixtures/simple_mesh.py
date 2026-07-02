import open3d as o3d


def make_plane():
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=10,
        height=10,
        depth=0.01,
    )
    mesh.translate([-5, -5, -0.01])
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene
