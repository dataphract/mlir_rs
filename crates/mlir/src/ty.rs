//! The MLIR type system.

use mlir_sys as ffi;

use crate::{is_fns, Type};

/// A trait for subtypes of [`Type`].
///
/// # Safety
///
/// Implementors of this trait must uphold the following invariants:
/// - `can_downcast` may return `true` only if the concrete type of `ty` is `Self`.
/// - `downcast_from` must return the same object passed in, and may only return `Ok` if
///   `can_downcast` returned `true` for `ty`.
pub unsafe trait TypeSubtype: Sized {
    /// Returns `true` if and only if `Self` is the concrete type of `ty`.
    fn can_downcast(ty: &Type) -> bool;

    /// Downcasts from `Type` to `Self` without checking invariants.
    ///
    /// # Safety
    ///
    /// This function is safe to call if and only if `ty` is the concrete type of `ty`.
    unsafe fn downcast_from_unchecked(ty: Type) -> Self;

    fn downcast_from(ty: Type) -> Result<Self, Type> {
        if Self::can_downcast(&ty) {
            Ok(unsafe { Self::downcast_from_unchecked(ty) })
        } else {
            Err(ty)
        }
    }
}

macro_rules! ty_types {
    ($(
        $v:vis struct $name:ident;
    )*) => {
        $(
            #[derive(Copy, Clone)]
            #[repr(transparent)]
            $v struct $name {
                inner: ffi::MlirType,
            }

            impl From<$name> for Type {
                fn from(other: $name) -> Type {
                    Type { inner: other.inner }
                }
            }

            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    Type::from(*self) == Type::from(*other)
                }
            }

            impl Eq for $name {}

            impl std::fmt::Display for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    std::fmt::Display::fmt(&Type::from(*self), f)
                }
            }

            impl $name {
                #[allow(dead_code)]
                pub(crate) unsafe fn from_raw(ty: ffi::MlirType) -> Option<$name> {
                    Some($name {
                        inner: ty,
                    })
                }

                #[allow(dead_code)]
                pub fn as_raw(&self) -> ffi::MlirType {
                    self.inner
                }
            }
        )*
    };
}

macro_rules! ty_downcast {
    ($($fn_name:ident => $subtype_name:ident),* $(,)?) => {
        $(
            unsafe impl TypeSubtype for $subtype_name {
                fn can_downcast(ty: &Type) -> bool {
                    Type::$fn_name(ty)
                }

                unsafe fn downcast_from_unchecked(ty: Type) -> Self {
                    Self::from_raw(ty.as_raw()).unwrap()
                }
            }
        )*
    };
}

ty_types! {
    pub struct BF16Type;
    pub struct F16Type;
    pub struct F32Type;
    pub struct F64Type;
    pub struct FunctionType;
}

is_fns! {
    impl Type {
        pub fn is_function = ffi::mlirTypeIsAFunction;
    }
}

ty_downcast! {
    is_function => FunctionType,
}

impl FunctionType {
    pub fn get(inputs: &[Type], results: &[Type]) -> FunctionType {
        crate::context().without_mutex(|cx| unsafe {
            let raw = ffi::mlirFunctionTypeGet(
                cx,
                inputs.len() as isize,
                inputs.as_ptr() as *const ffi::MlirType,
                results.len() as isize,
                results.as_ptr() as *const ffi::MlirType,
            );

            FunctionType::from_raw(raw).unwrap()
        })
    }
}
