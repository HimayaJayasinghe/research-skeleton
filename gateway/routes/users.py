"""
gateway/routes/users.py
User management endpoints: list, get, create, delete.
"""
from fastapi import APIRouter, HTTPException
from typing import List
from database.schemas import UserCreate, UserResponse, UserInDB
from database.crud import UserCRUD, FeatureProfileCRUD

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("/", response_model=List[UserResponse])
async def list_users():
    """List all enrolled users."""
    users = await UserCRUD.list_all()
    return [UserResponse(**u) for u in users]


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Get a specific user by ID."""
    user = await UserCRUD.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Also get feature profile
    profile = await FeatureProfileCRUD.get_by_user(user_id)
    return {
        "user": user,
        "feature_profile": profile,
    }


@router.post("/", response_model=UserResponse)
async def create_user(req: UserCreate):
    """Create a new user for enrollment."""
    # Check if name already exists
    existing = await UserCRUD.get_by_name(req.name)
    if existing:
        raise HTTPException(
            status_code=409, detail=f"User '{req.name}' already exists"
        )

    user = UserInDB(name=req.name, email=req.email)
    if req.notes:
        user.metadata["notes"] = req.notes

    await UserCRUD.create(user)
    return UserResponse(
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        enrollment_status=user.enrollment_status,
        enrollment_frames_count=user.enrollment_frames_count,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """Delete a user and their feature profile."""
    user_deleted = await UserCRUD.delete(user_id)
    profile_deleted = await FeatureProfileCRUD.delete_by_user(user_id)

    if not user_deleted:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "message": f"User {user_id} deleted",
        "profile_deleted": profile_deleted,
    }


@router.get("/count/total")
async def user_count():
    """Get total number of enrolled users."""
    count = await UserCRUD.count()
    return {"total_users": count}
