!/////////////////////////////////////////////////////////////////////////////////////////
!// Author: Manish Agarwal and Gourav Shrivastava  , IIT Delhi
!/////////////////////////////////////////////////////////////////////////////////////////

! Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.

module readdata
      contains
      subroutine readdcd(maxframes,maxatoms,x,y,z,xbox,ybox,zbox,natoms,nframes)
      integer i,j
      integer maxframes,maxatoms

      double precision d(6),xbox,ybox,zbox
      real*4, allocatable   :: x(:,:)
      real*4, allocatable   :: y(:,:)
      real*4, allocatable   :: z(:,:)

      real*4 dummyr
      integer*4 nset, natoms, dummyi,nframes,tframes
      character*4 dummyc
      
      open(10,file='../input/alk.traj.dcd',status='old',form='unformatted')
      read(10) dummyc, tframes,(dummyi,i=1,8),dummyr, (dummyi,i=1,9)
      read(10) dummyi, dummyr,dummyr
      read(10) natoms
      print*,"Total number of frames and atoms are",tframes,natoms

      allocate ( x(maxframes,natoms) )
      allocate ( y(maxframes,natoms) )
      allocate ( z(maxframes,natoms) )

      do i = 1,nframes
           read(10) (d(j),j=1, 6)
              
           read(10) (x(i,j),j=1,natoms)
           read(10) (y(i,j),j=1,natoms)
           read(10) (z(i,j),j=1,natoms)
      end do
      
      xbox=d(1)
      ybox=d(3)
      zbox=d(6)
      
      print*,"File reading is done: xbox,ybox,zbox",xbox,ybox,zbox
      return

      end subroutine readdcd
 end module readdata

program rdf
      use readdata
      use nvtx
      implicit none
      integer n,i,j,iconf,ind
      integer natoms,nframes,nbin
      integer maxframes,maxatoms
      parameter (maxframes=10,maxatoms=60000,nbin=2000)
      real*4, allocatable   :: x(:,:)
      real*4, allocatable   :: y(:,:)
      real*4, allocatable   :: z(:,:)
      double precision dx,dy,dz
      double precision xbox,ybox,zbox,cut
      double precision vol,r,del,s2,s2bond
      double precision, allocatable   ::  g(:)
      double precision rho,gr,lngr,lngrbond,pi,const,nideal,rf
      double precision rlower,rupper
      character  atmnm*4
      real*4 start,finish
      real*4 start_pair_gpu,finish_pair_gpu
        
      open(23,file='RDF.dat',status='unknown')
      open(24,file='Pair_entropy.dat',status='unknown')

      nframes=10
         
      call cpu_time(start)

      print*,"Going to read coordinates"
      call nvtxStartRange("Read File")
      call readdcd(maxframes,maxatoms,x,y,z,xbox,ybox,zbox,natoms,nframes)
      call nvtxEndRange

      allocate ( g(nbin) )
      g = 0.0d0
 
      pi=dacos(-1.0d0)
      vol=xbox*ybox*zbox
      rho=dble(natoms)/vol

      del=xbox/dble(2.0*nbin)
      write(*,*) "bin width is : ",del
      cut = dble(xbox * 0.5);

      !pair calculation
      call nvtxStartRange("Pair Calculation")
      call cpu_time(start_pair_gpu)
      do iconf=1,nframes
         if (mod(iconf,1).eq.0) print*,iconf
         do i=1,natoms
            do j=1,natoms
               dx=x(iconf,i)-x(iconf,j)
               dy=y(iconf,i)-y(iconf,j)
               dz=z(iconf,i)-z(iconf,j)

               dx=dx-nint(dx/xbox)*xbox
               dy=dy-nint(dy/ybox)*ybox
               dz=dz-nint(dz/zbox)*zbox
   
               r=dsqrt(dx**2+dy**2+dz**2)
               ind=int(r/del)+1
               !if (ind.le.nbin) then
               if(r<cut)then
                  !$acc atomic
                  g(ind)=g(ind)+1.0d0
               endif
            enddo
         enddo
      enddo
      call cpu_time(finish_pair_gpu)
      call nvtxEndRange
      
      !entropy calculation
      s2=0.01d0
      s2bond=0.01d0 
      const=(4.0d0/3.0d0)*pi*rho
      call nvtxStartRange("Entropy Calculation")
      do i=1,nbin
          rlower=dble((i-1)*del)
          rupper=rlower+del
          nideal=const*(rupper**3-rlower**3)
          g(i)=g(i)/(dble(nframes)*dble(natoms)*nideal)
          r=dble(i)*del
          if (r.lt.2.0) then
            gr=0.0
          else
            gr=g(i)
          endif

          if (gr.lt.1e-5) then
            lngr=0.0
          else
            lngr=dlog(gr)
          endif
          if (g(i).lt.1e-6) then
            lngrbond=0.01
          else
            lngrbond=dlog(g(i))
          endif

          s2=s2-2*pi*rho*((gr*lngr)-gr+1)*del*r**2.0
          s2bond=s2bond-2*pi*rho*((g(i)*lngrbond)-g(i)+1)*del*r*r

          
          rf=dble(i-.5)*del
          write(23,*) rf,g(i)
      enddo
      call nvtxEndRange

      write(24,*)"s2      : ",s2
      write(24,*)"s2bond  : ",s2bond
      call cpu_time(finish)
      print*,"Total Execution time: ",finish
      
      ! output timing into a file 
      open(1, file = 'result.txt', status = 'unknown', access= 'append')  
      !write(1,*) (finish_pair_gpu-start_pair_gpu) 
      write(1,*) (finish-start) 
      stop
      deallocate(x,y,z,g)
end
