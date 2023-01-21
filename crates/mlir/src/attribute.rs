use crate::{
    ffi,
    Attribute, Context, NamedAttribute, StringRef, Type,
};

pub const FUNCTION_TYPE_ATTR_NAME: &str = "function_type";
pub const FUNCTION_ARG_DICT_ATTR_NAME: &str = "arg_attrs";
pub const FUNCTION_RESULT_DICT_ATTR_NAME: &str = "res_attrs";

macro_rules! attr_types {
    ($(
        $v:vis struct $name:ident;
    )*) => {
        $(
            #[derive(Copy, Clone)]
            #[repr(transparent)]
            $v struct $name {
                inner: Attribute,
            }

            impl From<$name> for Attribute {
                fn from(other: $name) -> Attribute {
                    other.inner
                }
            }

            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    self.inner == other.inner
                }
            }

            impl Eq for $name {}

            impl std::fmt::Display for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    std::fmt::Display::fmt(&self.inner, f)
                }
            }

            impl $name {
                #[allow(dead_code)]
                pub(crate) unsafe fn from_raw(attr: ffi::MlirAttribute) -> Option<$name> {
                    Some($name {
                        inner: Attribute::from_raw(attr)?,
                    })
                }

                #[allow(dead_code)]
                pub fn as_raw(&self) -> ffi::MlirAttribute {
                    self.inner.inner
                }
            }
        )*
    };
}

macro_rules! attr_ctors {
    (
        $(
            $v:vis fn $name:ident::$fn_name:ident($(
                $arg:ident : $arg_ty:ty
            ),*) = $ctor_fn:path;
        )*
    ) => {
        $(
            impl $name {
                $v fn $fn_name($(
                    $arg : $arg_ty
                ),*) -> $name {
                    crate::context().without_mutex(|cx| unsafe {
                        $name::from_raw($ctor_fn(cx, $($arg.as_raw()),*)).unwrap()
                    })
                }
            }
        )*
    };
}
pub(crate) use attr_ctors;

macro_rules! attr_getters {
    ($(
        $v:vis fn $name:ident::$fn_name:ident(&self) -> $ret:ty = $getter_fn:path;
    )*) => {
        $(
            impl $name {
                $v fn $fn_name(&self) -> $ret {
                    unsafe {
                        <$ret>::from_raw($getter_fn(self.inner.inner))
                    }
                }
            }
        )*
    };
}

attr_types! {
    pub struct ArrayAttr;
    pub struct DictionaryAttr;
    pub struct FlatSymbolRefAttr;
    pub struct TypeAttr;
    // pub struct StringAttr;
}

attr_ctors! {
    pub fn FlatSymbolRefAttr::new( symbol: StringRef<'_>) = ffi::mlirFlatSymbolRefAttrGet;
}

attr_getters! {
    pub fn FlatSymbolRefAttr::value(&self) -> StringRef = ffi::mlirFlatSymbolRefAttrGetValue;
}

impl ArrayAttr {
    pub fn create(elements: &[Attribute]) -> ArrayAttr {
        crate::context()
            .without_mutex(|cx| unsafe {
                ArrayAttr::from_raw(ffi::mlirArrayAttrGet(
                    cx,
                    elements.len() as isize,
                    elements.as_ptr().cast(),
                ))
            })
            .unwrap()
    }
}

impl DictionaryAttr {
    pub fn create(elements: &[NamedAttribute]) -> DictionaryAttr {
        crate::context()
            .without_mutex(|cx| unsafe {
                DictionaryAttr::from_raw(ffi::mlirDictionaryAttrGet(
                    cx,
                    elements.len() as isize,
                    elements.as_ptr().cast(),
                ))
            })
            .unwrap()
    }
}

impl From<Type> for TypeAttr {
    fn from(value: Type) -> Self {
        TypeAttr::get(value)
    }
}

impl TypeAttr {
    pub fn get(ty: Type) -> TypeAttr {
        crate::context()
            .without_mutex(|cx| unsafe { TypeAttr::from_raw(ffi::mlirTypeAttrGet(ty.as_raw())) })
            .unwrap()
    }
}
