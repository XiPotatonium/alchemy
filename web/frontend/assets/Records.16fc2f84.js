var $=Object.defineProperty;var x=(e,l,t)=>l in e?$(e,l,{enumerable:!0,configurable:!0,writable:!0,value:t}):e[l]=t;var b=(e,l,t)=>(x(e,typeof l!="symbol"?l+"":l,t),t);import{c as A,d as R,m as T,a as F,u as j,b as V,e as K,f as I,g as i,h as M,I as N,i as E,j as Q,k as z,t as C,l as G,n as H,p as J,V as U,o as L,F as P,q as Y,r as q,s as _,v as O,w as B,x as v,y as W,z as X,A as y,B as Z,C as ee,D as te,E as ae,G as le,H as se,J as ie,K as S,L as ne,M as w,N as D}from"./index.93fee454.js";import{V as re}from"./VToolbar.fcd4242e.js";const oe=A("v-breadcrumbs-divider","li"),ce=R({name:"VBreadcrumbsItem",props:{active:Boolean,activeClass:String,activeColor:String,color:String,disabled:Boolean,title:String,...T(),...F({tag:"li"})},setup(e,l){let{slots:t,attrs:f}=l;const o=j(e,f),m=V(()=>{var n;return e.active||((n=o.isActive)==null?void 0:n.value)}),h=V(()=>m.value?e.activeColor:e.color),{textColorClasses:u,textColorStyles:g}=K(h);return I(()=>{var n;const d=o.isLink.value?"a":e.tag;return i(d,{class:["v-breadcrumbs-item",{"v-breadcrumbs-item--active":m.value,"v-breadcrumbs-item--disabled":e.disabled,"v-breadcrumbs-item--link":o.isLink.value,[`${e.activeClass}`]:m.value&&e.activeClass},u.value],style:[g.value],href:o.href.value,"aria-current":m.value?"page":void 0,onClick:o.navigate},{default:()=>{var s;return[(s=(n=t.default)==null?void 0:n.call(t))!=null?s:e.title]}})}),{}}}),ue=M()({name:"VBreadcrumbs",props:{activeClass:String,activeColor:String,bgColor:String,color:String,disabled:Boolean,divider:{type:String,default:"/"},icon:N,items:{type:Array,default:()=>[]},...E(),...Q(),...F({tag:"ul"})},setup(e,l){let{slots:t}=l;const{backgroundColorClasses:f,backgroundColorStyles:o}=z(C(e,"bgColor")),{densityClasses:m}=G(e),{roundedClasses:h}=H(e);return J({VBreadcrumbsItem:{activeClass:C(e,"activeClass"),activeColor:C(e,"activeColor"),color:C(e,"color"),disabled:C(e,"disabled")}}),I(()=>{var u;const g=!!(t.prepend||e.icon);return i(e.tag,{class:["v-breadcrumbs",f.value,m.value,h.value],style:o.value},{default:()=>[g&&i(U,{key:"prepend",defaults:{VIcon:{icon:e.icon,start:!0}}},{default:()=>[i("div",{class:"v-breadcrumbs__prepend"},[t.prepend?t.prepend():e.icon&&i(L,null,null)])]}),e.items.map((n,d,s)=>{var r;return i(P,null,[i(ce,Y({key:d,disabled:d>=s.length-1},typeof n=="string"?{title:n}:n),{default:t.title?()=>{var a;return(a=t.title)==null?void 0:a.call(t,{item:n,index:d})}:void 0}),d<s.length-1&&i(oe,null,{default:()=>{var a;return[(a=(r=t.divider)==null?void 0:r.call(t,{item:n,index:d}))!=null?a:e.divider]}})])}),(u=t.default)==null?void 0:u.call(t)]})}),{}}}),fe=q({__name:"Records",setup(e){const l=_(["records"]);class t{constructor(r,a,p,c){b(this,"ty");b(this,"icon");b(this,"path");b(this,"ctime");b(this,"mtime");this.ty=r,r==="folder"?this.icon="mdi-folder":this.icon="mdi-file",this.path=a,this.ctime=p,this.mtime=c}get is_folder(){return this.ty==="folder"}get name(){return this.path[this.path.length-1]}get ctime_str(){return this.ctime.toLocaleString()}get mtime_str(){return this.mtime.toLocaleString()}}const f=_([]);async function o(){var s=l.value.slice(1).join("/");await X.get("/api/lsFiles/"+s).then(r=>{let a=Array.from(r.data.subitems.map(c=>new t(c.ty,c.path,c.ctime,c.mtime)));a.sort((c,k)=>c.is_folder&&!k.is_folder?-1:k.is_folder&&!c.is_folder?1:c.name.localeCompare(k.name)),f.value=a;var p=r.data.path;console.assert(p.join("/")===s)})}async function m(s){s.is_folder?(l.value.push(s.name),await o()):window.open("/api/getFile/"+l.value.slice(1).concat(s.name).join("/"))}async function h(){l.value.pop(),await o()}const u=_(""),g=_("");function n(){console.log("You search "+u.value),g.value=u.value}function d(s,r){return s.filter(a=>r?a.name.includes(r):!0)}return O(o),(s,r)=>(y(),B(W,{fluid:""},{default:v(()=>[i(re,{align:"center"},{default:v(()=>[i(ue,{items:l.value},{prepend:v(()=>[i(Z,{variant:"text",icon:"mdi-arrow-left",disabled:l.value.length==1,onClick:h},null,8,["disabled"])]),_:1},8,["items"]),i(ee),i(te,{clearable:"","hide-details":"","single-line":"","append-inner-icon":"mdi-magnify",modelValue:u.value,"onUpdate:modelValue":r[0]||(r[0]=a=>u.value=a),onKeydown:ae(S(n,["prevent"]),["enter"]),"onClick:clear":n},null,8,["modelValue","onKeydown"])]),_:1}),i(ie,null,{default:v(()=>[(y(!0),le(P,null,se(d(f.value,g.value),a=>(y(),B(ne,{density:"compact",key:a.name,title:a.name,onDblclick:S(p=>m(a),["prevent"])},{prepend:v(()=>[i(L,null,{default:v(()=>[w(D(a.icon),1)]),_:2},1024)]),append:v(()=>[w(D(a.mtime_str),1)]),_:2},1032,["title","onDblclick"]))),128))]),_:1})]),_:1}))}});export{fe as default};