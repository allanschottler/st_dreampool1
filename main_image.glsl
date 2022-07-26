//==========================================================================//
// Dream Pool #1                                                            //
//                                                                          //
// https://www.shadertoy.com/view/wtXXWS                                    //
//                                                                          //
//==========================================================================//

// Boost this if you've got the pc for it
#define AA 1

#define MAX_STEPS 100
#define MAX_DIST  200.
#define SURF_DIST .01
#define ID_WATER  0.
#define ID_FLOOR  1.
#define ID_WALL   2.
//#define FULLBRIGHT

const float waterLevel = 2.5;
const float wallDistance = 18.1;
const float camDistance = 1.9;
const float tunnelHeight = 5.9;
const float tunnelsSpacing = 30.;
const float pillarsSpacing = 15.;
const float pillarsDistance = 20.;
const vec3 lightAmbient = vec3(244./255., 233./255., 155./255.)*.3;//vec3(.29,.25,.2)*1.;
//const vec3 light0 = vec3(-6., 15., -10.)*10.;
const vec3 sunDir = vec3(-4., 15., -10.);
const vec3 waterColor = vec3(0.,.6,.6);
const vec3 tileColor = vec3(0.84, 0.88, 0.89);
const vec3 cementColor = vec3(0.24, 0.28, 0.19);
const vec2 camMinMaxHeight = vec2(4.3+0.8*2.7,2.7);
const vec3 camPos = vec3(0.,camMinMaxHeight.x,-camDistance);

float time = 0.; 

//==========================================================================//
// Auxilary functions                                                       //
//==========================================================================//

float sfract(float n)
{ return smoothstep(0.0,1.0,fract(n)); }

float rand(vec2 n)
{ return fract(abs(sin(dot(n,vec2(5.3357,-5.8464))))*256.75+0.325); }

float noise(vec2 n)
{
    n *= .5;
    float h1 = mix(rand(vec2(floor(n.x),floor(n.y))),
                   rand(vec2(ceil(n.x),floor(n.y))),
                   sfract(n.x));
    float h2 = mix(rand(vec2(floor(n.x),ceil(n.y))),
                   rand(vec2(ceil(n.x),ceil(n.y))),
                   sfract(n.x));
    return mix(h1,h2,sfract(n.y));
}

mat2 rot(float a) 
{ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

//==========================================================================//
// Signed distance functions                                                //
//==========================================================================//

float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdLink(vec3 p, float le, float r1, float r2)
{
    vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
}

float sdCappedCylinder(vec3 p, float h, float r)
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdLip(vec3 p)
{
    const float le = 6., r1 = 4., r2 = .75;
    const float off = tunnelsSpacing;
    p.x = mod(p.x+off,2.*off)-off;
    p.y = max(p.y, -0.25);
    p.z += le;
    p.y += r2;    
    p = p.xzy;
    return sdLink(p,le,r1,r2);
}

float sdSteps(vec3 p)
{
    const float off = tunnelsSpacing;
    p.x = mod(p.x+off,2.*off)-off;
    p.z -= wallDistance;
    p.z = min(p.z, 1.);
    return min(min(
        sdCappedCylinder(p, 5., 1.5),
        sdCappedCylinder(p, 6., 1.)),
        sdCappedCylinder(p, 7., .5));
}

float sdPillars(vec3 p)
{
    const float off = pillarsSpacing;
    p.z += pillarsDistance;
    p.x = mod(p.x+off,2.*off)-off;
    return min(
        sdCappedCylinder(p,5.,200.),
        sdBox(p, vec3(off, 3., 7.)));
}

float sdWater(vec3 p)
{
    //Distort coordinates
    p += vec3(iTime*.2,0.0,0.0)+vec3(noise(p.xz),0.0,noise(p.xz+8.0))*.2;
    float height = 0.01*pow(noise(p.xz+vec2(iTime*0.7,iTime*0.6))*0.5
                           +noise(p.xz*8.0+vec2(iTime))*0.35
                           +noise(p.xz*16.0+vec2(0.0,iTime*0.5))*0.1
                           +noise(p.xz*24.0)*0.05,0.25);
    float model = p.y-height*20.;
    return .8*model;
}

float sdTiles(vec3 p)
{
    float planeDist = p.y;
    float panel = 0.;
    float fade = 2.-min(length(p),8.)/16.;
    if(fade>0.)
    {	
        panel = max(0.9,max(abs(-1.+2.*mod(p.x,.5)),
                            abs(-1.+2.*mod(p.z,.5))))-.9;
    	panel = panel*panel;
        //panel += .25*noise(p.xz);
     	planeDist = mix(p.y,p.y+.95,panel*fade);
    }
    return planeDist;
}

float sdFloor(vec3 p)
{
    return sdTiles(p);
    //return max(-sdHoles(p), sdTiles(p));
}

float sdTunnel(vec3 p, vec3 bP, float off)
{
    const float offX = tunnelsSpacing;
    vec3 ogP = p;
    p.x = mod(p.x+offX,2.*offX)-offX;
    bP.x = mod(p.x+offX,2.*offX)-offX;
    
    return max(-min(
        sdBox(bP, vec3(4., off, 12.5)),
        sdCappedCylinder(p, 4., 12.5)),
        sdTiles(ogP));
}

float sdWall(vec3 p)
{
    const float off = tunnelHeight;
    const float off2 = tunnelsSpacing;
    
    p.z -= wallDistance; //push entire wall back  
    
    vec3 bP = p;
    
    p.z *= -1.;     //flip towards camera
    p.y -= 2.*off;  //shift upwards
    p = p.xzy;      //rotate
    
    vec3 lP = p;
    lP.x = mod(p.x+off2,2.*off2)-off2;
    
    bP.y -= off;    //shift upwards
    
    return min(sdTunnel(p, bP, off), sdLip(lP));
}

vec2 sdWorldNoWater(vec3 p)
{
    float id = ID_FLOOR;
    float d = min(min(min(
        sdWall(p),
        sdFloor(p)),
        sdSteps(p)),
        sdPillars(p));
    return vec2(d, id);
}

vec2 sdWorld(vec3 p)
{   
    vec2 di = sdWorldNoWater(p);
    float d = di.x;
    float id = di.y; 
    float worldDist = d;
    
    p.y -= waterLevel;
    d = min(d,sdWater(p));
    
    if(d < worldDist)
        id = ID_WATER;
        
    return vec2(d, id);
}

//==========================================================================//
// Marching functions                                                       //
//==========================================================================//

vec2 rayMarch(vec3 ro, vec3 rd)
{
    float dO =0.;
    vec2 obj = vec2(-1.);
    
    for(int i=0; i<MAX_STEPS; i++)
    {
		vec3 p = ro + rd*dO;
        obj = sdWorld(p);
        float dS = obj.x;
        dO += dS;
        if(dO>MAX_DIST || dS<SURF_DIST) 
            break;        
    }
    
    return vec2(dO, obj.y);
}

vec2 refractMarch(vec3 ro, vec3 rd)
{
    float dO =0.;
    vec2 obj = vec2(-1.);
    
    for(int i=0; i<MAX_STEPS; i++)
    {
		vec3 p = ro + rd*dO;
        obj = sdWorldNoWater(p);
        float dS = obj.x;
        dO += dS;
        if(dO>MAX_DIST || dS<SURF_DIST) 
            break;        
    }
    
    return vec2(dO, obj.y);
}

float shadowMarch(vec3 ro, vec3 rd)
{
    float dO =0.;
    float res = 1.;
    float dS =0.;
    for(int i=0; i<MAX_STEPS; i++)
    {
        /*if(dO>MAX_DIST)
            break;*/
		vec3 p = ro + rd*dO;
        vec2 obj = sdWorldNoWater(p);
        dS = obj.x;
        dO += dS;
        if(dS<SURF_DIST) 
            return 0.;
        res = min(res, 8.*dS/dO);
    }
         
    return res;
}

//==========================================================================//
// Computes normal at given point, with and without water.                  //
//==========================================================================//

vec4 getNormal(vec3 p)
{
    vec2 item = sdWorld(p);
    float d = item.x;
    
    vec2 e = vec2(.01, 0.);
    vec3 n = d - vec3(	sdWorld(p-e.xyy).x,
                 		sdWorld(p-e.yxy).x,
                 		sdWorld(p-e.yyx).x);
    return vec4(normalize(n), item.y);
}

vec4 getNormalNoWater(vec3 p)
{
    vec2 item = sdWorldNoWater(p);
    float d = item.x;
    
    vec2 e = vec2(.01, 0.);
    vec3 n = d - vec3(	sdWorldNoWater(p-e.xyy).x,
                 		sdWorldNoWater(p-e.yxy).x,
                 		sdWorldNoWater(p-e.yyx).x);
    return vec4(normalize(n), item.y);
}

//==========================================================================//
// Computes light emission from given point                                 //
//==========================================================================//

/*float caustics(vec3 p) 
{
    //vec3 lp = light0;
    vec3 ray = normalize(sunDir);

    /*vec2 shadow = scene(lp, ray);
    float l = distance(lp + ray * shadow.x, p);
    if (l > 0.01) {
        return 0.0;
    }*/

    /*vec2 d = rayMarch(lp, ray);
    vec3 waterSurface = lp + ray * d.x;

    vec3 refractRay = reflect(ray, vec3(0., 1., 0.));
    float beforeHit = refractMarch(waterSurface, refractRay).x;
    vec3 beforePos = waterSurface + refractRay * beforeHit;

    //vec3 noisePos = waterSurface + vec3(0.,iTime * 2.0,0.);
    //float height = noise(noisePos);
    //vec3 deformedWaterSurface = waterSurface + vec3(0., height, 0.);
    vec3 deformedWaterSurface = waterSurface;
    
    refractRay = reflect(ray, getNormal(p).xyz);
    float afterHit = refractMarch(deformedWaterSurface, refractRay).x;
    vec3 afterPos = deformedWaterSurface + refractRay * afterHit;

    float beforeArea = length(dFdx(beforePos)) * length(dFdy(beforePos));
    float afterArea = length(dFdx(afterPos)) * length(dFdy(afterPos));
    return max(beforeArea / afterArea, .001);
}
*/

vec3 getLight(vec3 p, vec3 rd)
{
    vec3 col = vec3(1.);
    vec4 item = getNormal(p);
    vec3 n = item.xyz;
    vec3 light = sunDir;
    //light.x += cos(time*.01);
    vec3 l = normalize(light); 
    //l = normalize(lightPos-p);
    float dif = dot(n,l);
    float sub = max(0.,.5+.5*dif);
    dif = max(0.,dif);
    
    if(item.w == ID_WATER)
    {        
        //col*=vec3(.71, 0.9, 0.92);
        //col += pow(sub,2.)*vec3(0.95,0.61,0.38);
        //float c = caustics(p) * 0.6;
        //col = vec3(c,c,c); 
        col = waterColor;
        col += pow(sub,2.)*vec3(0.95,0.61,0.38);
        
        //XZ
        float corr = 4.5;
        vec3 v1 = normalize(vec3(sunDir.x,0.,sunDir.z));
        vec3 p1 = vec3((pillarsSpacing)*4.-corr,0.,pillarsDistance);
        vec3 p2 = vec3(-(pillarsSpacing)*2.+corr,0.,pillarsDistance);
        vec3 l2 = cross(vec3(0.,1.,0.),v1);
        vec3 l1 = cross(v1,vec3(0.,1.,0.));
        
        float fade = length(p-camPos);
        float d1 = dot(p1,l1);
        float d2 = dot(p2,l2);
        
        if(dot(vec4(l1,d1), vec4(p,1.)) > 0. && 
           dot(vec4(l2,d2), vec4(p,1.)) > 0.)
        {
            float fade = .92;//*clamp(0.,1.,1.-fade/50.);//0.92*(1.-fade/50.));
            vec3 nrd = refract(rd, n, fade);
            //vec3 nrd3 = reflect(rd, n);

            vec2 dr = refractMarch(p, nrd);
            p = p + nrd*dr.x;
            rd = nrd;   
        }
    }    
        
    item = getNormalNoWater(p);
    n = item.xyz;

    dif = dot(n,l);
    //sub = max(0.,.5+.5*dif);
    dif = max(0.,dif);

    float d = shadowMarch(p+n*SURF_DIST*2., l);
    d = sqrt(d);
    dif *=d;
    #ifndef FULLBRIGHT
    col *= vec3(dif)+(vec3(.6,.1,.0)-vec3(dif))*(d*.1);
    #endif
    vec2 ll = mod(p.xz,.5);
    col *= mix(tileColor,cementColor,ll.x>.46 || ll.y>.46?1.0:0.);

    if(dif>0.)
    {
        float spec = pow( max(0.,dot(l, reflect(rd, n))), 100.);
        col +=vec3(.45*spec);
    }
    
    return col;
}

//==========================================================================//
// Main ray marching function. Computes color of current pixel              //
//==========================================================================//

vec3 render(in vec3 ro, in vec3 rd)
{
    vec2 d = rayMarch(ro, rd);
    vec3 p = ro + rd*d.x;
    vec3 dif = getLight(p, rd);
    
    return mix(dif,lightAmbient,pow(min(1.,d.x*.005),.5));
}

//==========================================================================//
// Main                                                                     //
//==========================================================================//

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float mTime = iTime;//+ mouse.x*8.;
    
    vec3 tot = vec3(0.0);
    
    //anti aliasing
    #if AA>1
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ ) {        
    vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
    vec2 uv = 0.5*(-iResolution.xy + 2.0*(fragCoord+o))/iResolution.y;

    uv *=1.+.2*pow(dot(uv,uv),2.);      
    float ad = 0.5*sin(fragCoord.x*147.0)*sin(fragCoord.y*131.0);
    time = mTime - 0.5*(1.0/30.0)*(float(m*AA+n)+ad)/float(AA*AA-1);
    #else    
    vec2 uv = 0.5*(-iResolution.xy + 2.0*fragCoord)/iResolution.y;
    uv *=1.+.2*pow(dot(uv,uv),2.);
    time = mTime;
    #endif  
        
    vec2 mouse = iMouse.xy;
    
    //camera
    vec2 delta = (iMouse.xy-iResolution.xy/2.)/iResolution.x;
    delta.x *= 2.;
    if(delta.y > 0.) delta.y *= 2.;
    delta.y *= -1.;
    vec2 camrot = delta.yx;
    
    vec4 pitch = vec4(cos(camrot.x),sin(camrot.x)*vec2(1,-1),0);
    vec4 yaw = vec4(cos(camrot.y),sin(camrot.y)*vec2(1,-1),1);
    vec3 i = vec3(yaw.x,0,yaw.z);
    vec3 j = pitch.yxy*yaw.ywx;
    vec3 k = pitch.xzx*yaw.ywx;
    
    vec2 mm = camMinMaxHeight;
    //vec3 ro = vec3(0.,mm.x+mouse.y*mm.y,-camDistance);
    vec3 ro = camPos;
    vec3 rd = normalize(vec3(uv.x, uv.y-.2,1.));
    
    rd = rd.x*i + rd.y*j + rd.z*k;
    //pos = pos.x*cam.i + pos.y*cam.j + cam.pos;
        
    //renders
	  tot += render(ro, rd);        
        
    #if AA>1
    } tot /= float(AA*AA);    
    #endif
    
    // tone mapping			
    tot = tot*1.2/(1.0+tot);

	  // gamma	
	  tot = pow( clamp(tot,0.0,1.0), vec3(0.45) );

    tot = tot*tot*(3.0-2.0*tot);
    
	  // vignetting	
	  vec2 q = fragCoord.xy / iResolution.xy;
	  tot *= 0.5 + 0.5*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1 );
    
    fragColor = vec4(tot, 1.);
}
